#!/usr/bin/env python3
"""Simple multihost runner for TPU pods.

Runs a command on all workers of a TPU pod simultaneously.

Example usage:
  python3 multihost_runner.py --tpu-name=my-tpu-v5e-16 --command="bash script.sh"

  Or with script directory copying:
  python3 multihost_runner.py --tpu-name=my-tpu-v5e-16 --command="bash script.sh" --script-dir=.
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime


def get_project():
    """Get current gcloud project."""
    result = subprocess.run(
        ["gcloud", "config", "get", "project"],
        check=True,
        capture_output=True,
        text=True
    )
    project = result.stdout.strip()
    if not project:
        sys.exit("Error: No project set. Run: gcloud config set project <project>")
    return project


def get_zone():
    """Get current gcloud zone."""
    result = subprocess.run(
        ["gcloud", "config", "get", "compute/zone"],
        check=True,
        capture_output=True,
        text=True
    )
    zone = result.stdout.strip()
    if not zone:
        sys.exit("Error: No zone set. Run: gcloud config set compute/zone <zone>")
    return zone


def get_num_workers(tpu_name, project, zone):
    """Get number of workers in the TPU."""
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "describe", tpu_name,
        "--flatten=networkEndpoints[]",
        "--format=csv[no-heading](networkEndpoints.ipAddress)",
        f"--project={project}",
        f"--zone={zone}"
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Count number of IP addresses (one per worker)
        num_workers = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        if num_workers == 0:
            sys.exit(f"Error: TPU {tpu_name} has no workers")
        return num_workers
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: Failed to describe TPU {tpu_name}: {e.stderr}")
    except ValueError:
        sys.exit(f"Error: Could not parse number of workers from gcloud output")


def scp_to_workers(tpu_name, script_dir, project, zone, num_workers):
    """Copy script directory to all workers."""
    print(f"Copying {script_dir} to {num_workers} workers...")

    # Create tarball, excluding tpu_inference package to avoid shadowing installed vllm-tpu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_name = f"script_dir_{timestamp}.tar.gz"
    tar_path = f"/tmp/{tar_name}"

    subprocess.run(
        ["tar", "-czf", tar_path,
         "--exclude=tpu_inference", "--exclude=.git", "--exclude=__pycache__",
         "--exclude=*.pyc", "--exclude=.pytest_cache",
         "-C", script_dir, "."],
        check=True
    )

    # SCP to all workers with small delays to avoid rate limiting
    # GCP has 100 mutations/minute limit for TPU API
    processes = []
    for worker in range(num_workers):
        cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "scp",
            f"--worker={worker}",
            tar_path,
            f"{tpu_name}:~/",
            "--strict-host-key-checking=no",
            f"--project={project}",
            f"--zone={zone}",
            "--quiet"
        ]
        processes.append(subprocess.Popen(cmd))
        # Small delay between launching SCPs to avoid rate limiting
        if worker < num_workers - 1:
            time.sleep(1)

    # Wait for all SCPs to complete
    return_codes = []
    for p in processes:
        return_codes.append(p.wait())

    # Check if any SCP failed
    if any(rc != 0 for rc in return_codes):
        failed_workers = [i for i, rc in enumerate(return_codes) if rc != 0]
        print(f"Error: SCP failed for workers {failed_workers}")
        sys.exit(1)

    # Cleanup
    os.remove(tar_path)

    return tar_name


def run_command_on_workers(tpu_name, command, project, zone, num_workers, tar_name=None):
    """Run command on all workers simultaneously."""
    print(f"Running command on {num_workers} workers...")

    # Build remote command
    if tar_name:
        # Extract tarball and run command
        remote_cmd = f"mkdir -p ~/work-dir && cd ~/work-dir && tar -xzf ~/{tar_name} && {command}"
    else:
        remote_cmd = command

    # SSH to all workers with small delays to avoid rate limiting
    processes = []
    log_files = []

    for worker in range(num_workers):
        log_file = f"/tmp/multihost_worker_{worker}.log"
        log_files.append(log_file)

        cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name,
            f"--worker={worker}",
            "--command", remote_cmd,
            "--strict-host-key-checking=no",
            f"--project={project}",
            f"--zone={zone}",
            "--quiet"
        ]

        # Worker 0 outputs to stdout, others to log files
        if worker == 0:
            log = None  # stdout
        else:
            log = open(log_file, "w")

        processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        # Small delay between launching SSHs to avoid rate limiting
        if worker < num_workers - 1:
            time.sleep(1)

    # Monitor progress
    start_time = time.time()
    while True:
        statuses = [p.poll() for p in processes]
        completed = sum(1 for s in statuses if s is not None)

        if completed == num_workers:
            break

        elapsed = time.time() - start_time
        print(f"[{elapsed:.1f}s] {completed}/{num_workers} workers completed...", end='\r')
        time.sleep(1)

    print()  # newline after progress

    # Check return codes
    return_codes = [p.poll() for p in processes]
    max_return = max(return_codes)

    if max_return != 0:
        failed = [i for i, rc in enumerate(return_codes) if rc != 0]
        print(f"Error: Workers {failed} failed with return codes {[return_codes[i] for i in failed]}")
        print("Check logs:")
        for worker in failed:
            if worker > 0:  # Worker 0 already printed to stdout
                print(f"  - {log_files[worker]}")
        return max_return

    print("All workers completed successfully!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run commands on TPU pod workers")
    parser.add_argument("--tpu-name", required=True, help="TPU name")
    parser.add_argument("--command", required=True, help="Command to run on all workers")
    parser.add_argument("--script-dir", default=None, help="Directory to copy to workers (optional)")
    parser.add_argument("--project", default=None, help="GCP project (defaults to gcloud config)")
    parser.add_argument("--zone", default=None, help="GCP zone (defaults to gcloud config)")

    args = parser.parse_args()

    # Get project and zone
    project = args.project or get_project()
    zone = args.zone or get_zone()

    print(f"TPU: {args.tpu_name}")
    print(f"Project: {project}")
    print(f"Zone: {zone}")
    print()

    # Get number of workers
    num_workers = get_num_workers(args.tpu_name, project, zone)
    print(f"Found {num_workers} workers")
    print()

    # Copy script directory if provided
    tar_name = None
    if args.script_dir:
        if not os.path.isdir(args.script_dir):
            sys.exit(f"Error: Script directory {args.script_dir} does not exist")
        tar_name = scp_to_workers(args.tpu_name, args.script_dir, project, zone, num_workers)
        print()

    # Run command
    return_code = run_command_on_workers(args.tpu_name, args.command, project, zone, num_workers, tar_name)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
