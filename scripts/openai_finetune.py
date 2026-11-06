#!/usr/bin/env python3
"""
OpenAI Fine-Tuning API Script

Programmatically upload training data and create fine-tuning jobs via OpenAI API.

Usage:
    # Create a new fine-tuning job
    python openai_finetune.py create --file path/to/training.jsonl --model gpt-4o-mini-2024-07-18

    # Create with custom suffix (appears in model name)
    python openai_finetune.py create --file data.jsonl --model gpt-4o-mini-2024-07-18 --suffix my-experiment

    # Create with validation file
    python openai_finetune.py create --file train.jsonl --model gpt-4o-mini-2024-07-18 --validation-file val.jsonl

    # List recent fine-tuning jobs
    python openai_finetune.py list

    # Check status of a specific job
    python openai_finetune.py status --job-id ftjob-abc123

    # Monitor a job until completion
    python openai_finetune.py monitor --job-id ftjob-abc123

    # Cancel a job
    python openai_finetune.py cancel --job-id ftjob-abc123

    # List uploaded files
    python openai_finetune.py files
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    """Initialize OpenAI client from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Add it to your .env file or export it.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def upload_file(client: OpenAI, file_path: str, purpose: str = "fine-tune") -> str:
    """Upload a file to OpenAI and return the file ID."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Uploading {file_path}...")
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)

    print(f"  Uploaded: {response.id} ({response.filename})")
    return response.id


def create_finetune_job(
    client: OpenAI,
    training_file: str,
    model: str,
    validation_file: str | None = None,
    suffix: str | None = None,
    n_epochs: int | None = None,
    batch_size: int | str | None = None,
    learning_rate_multiplier: float | None = None,
) -> str:
    """Create a fine-tuning job and return the job ID."""
    # Upload training file
    training_file_id = upload_file(client, training_file)

    # Upload validation file if provided
    validation_file_id = None
    if validation_file:
        validation_file_id = upload_file(client, validation_file)

    # Build hyperparameters
    hyperparameters = {}
    if n_epochs is not None:
        hyperparameters["n_epochs"] = n_epochs
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

    # Create job
    print(f"\nCreating fine-tuning job...")
    print(f"  Model: {model}")
    print(f"  Training file: {training_file_id}")
    if validation_file_id:
        print(f"  Validation file: {validation_file_id}")
    if suffix:
        print(f"  Suffix: {suffix}")
    if hyperparameters:
        print(f"  Hyperparameters: {hyperparameters}")

    kwargs = {
        "training_file": training_file_id,
        "model": model,
    }
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    if suffix:
        kwargs["suffix"] = suffix
    if hyperparameters:
        kwargs["hyperparameters"] = hyperparameters

    job = client.fine_tuning.jobs.create(**kwargs)

    print(f"\nJob created successfully!")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"\nMonitor with: python openai_finetune.py monitor --job-id {job.id}")

    return job.id


def list_jobs(client: OpenAI, limit: int = 10):
    """List recent fine-tuning jobs."""
    print(f"Recent fine-tuning jobs (limit={limit}):\n")
    jobs = client.fine_tuning.jobs.list(limit=limit)

    if not jobs.data:
        print("  No fine-tuning jobs found.")
        return

    for job in jobs.data:
        status_emoji = {
            "succeeded": "✓",
            "failed": "✗",
            "cancelled": "⊘",
            "running": "⟳",
            "queued": "◷",
            "validating_files": "◷",
        }.get(job.status, "?")

        model_info = f" → {job.fine_tuned_model}" if job.fine_tuned_model else ""
        print(f"  [{status_emoji}] {job.id} | {job.status} | {job.model}{model_info}")


def get_job_status(client: OpenAI, job_id: str):
    """Get detailed status of a fine-tuning job."""
    job = client.fine_tuning.jobs.retrieve(job_id)

    print(f"Job: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Model: {job.model}")
    print(f"  Created: {job.created_at}")

    if job.finished_at:
        print(f"  Finished: {job.finished_at}")
    if job.fine_tuned_model:
        print(f"  Fine-tuned model: {job.fine_tuned_model}")
    if job.trained_tokens:
        print(f"  Trained tokens: {job.trained_tokens:,}")
    if job.error:
        print(f"  Error: {job.error}")

    # Get recent events
    print("\nRecent events:")
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
    for event in reversed(events.data):
        print(f"  [{event.created_at}] {event.message}")


def monitor_job(client: OpenAI, job_id: str, poll_interval: int = 30):
    """Monitor a job until completion."""
    print(f"Monitoring job {job_id} (polling every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring.\n")

    seen_events = set()

    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)

            # Print new events
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=20
            )
            for event in reversed(events.data):
                if event.id not in seen_events:
                    print(f"[{event.created_at}] {event.message}")
                    seen_events.add(event.id)

            # Check if done
            if job.status in ("succeeded", "failed", "cancelled"):
                print(f"\nJob finished with status: {job.status}")
                if job.fine_tuned_model:
                    print(f"Fine-tuned model: {job.fine_tuned_model}")
                if job.error:
                    print(f"Error: {job.error}")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Job is still running.")
        print(f"Resume with: python openai_finetune.py monitor --job-id {job_id}")


def cancel_job(client: OpenAI, job_id: str):
    """Cancel a fine-tuning job."""
    print(f"Cancelling job {job_id}...")
    job = client.fine_tuning.jobs.cancel(job_id)
    print(f"Job status: {job.status}")


def list_files(client: OpenAI, purpose: str | None = "fine-tune"):
    """List uploaded files."""
    print(f"Uploaded files (purpose={purpose or 'all'}):\n")
    files = client.files.list(purpose=purpose) if purpose else client.files.list()

    if not files.data:
        print("  No files found.")
        return

    for f in files.data:
        size_kb = f.bytes / 1024 if f.bytes else 0
        print(f"  {f.id} | {f.filename} | {size_kb:.1f} KB | {f.purpose}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Fine-Tuning API Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # create command
    create_parser = subparsers.add_parser("create", help="Create a fine-tuning job")
    create_parser.add_argument(
        "--file", "-f", required=True, help="Path to training JSONL file"
    )
    create_parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-2024-08-06",
        help="Base model to fine-tune",
    )
    create_parser.add_argument(
        "--validation-file", "-v", help="Path to validation JSONL file (optional)"
    )
    create_parser.add_argument(
        "--suffix", "-s", help="Suffix for the fine-tuned model name (optional)"
    )
    create_parser.add_argument(
        "--epochs", "-e", type=int, help="Number of epochs (optional, auto by default)"
    )
    create_parser.add_argument(
        "--batch-size",
        "-b",
        help="Batch size (integer or 'auto', optional)",
    )
    create_parser.add_argument(
        "--learning-rate",
        "-l",
        type=float,
        help="Learning rate multiplier (optional)",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List recent fine-tuning jobs")
    list_parser.add_argument(
        "--limit", "-n", type=int, default=10, help="Number of jobs to list"
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("--job-id", "-j", required=True, help="Job ID")

    # monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor job until completion"
    )
    monitor_parser.add_argument("--job-id", "-j", required=True, help="Job ID")
    monitor_parser.add_argument(
        "--interval", "-i", type=int, default=30, help="Poll interval in seconds"
    )

    # cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("--job-id", "-j", required=True, help="Job ID")

    # files command
    files_parser = subparsers.add_parser("files", help="List uploaded files")
    files_parser.add_argument(
        "--all", "-a", action="store_true", help="Show all files, not just fine-tune"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = get_client()

    if args.command == "create":
        batch_size = args.batch_size
        if batch_size and batch_size != "auto":
            batch_size = int(batch_size)
        create_finetune_job(
            client,
            training_file=args.file,
            model=args.model,
            validation_file=args.validation_file,
            suffix=args.suffix,
            n_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate_multiplier=args.learning_rate,
        )
    elif args.command == "list":
        list_jobs(client, limit=args.limit)
    elif args.command == "status":
        get_job_status(client, args.job_id)
    elif args.command == "monitor":
        monitor_job(client, args.job_id, poll_interval=args.interval)
    elif args.command == "cancel":
        cancel_job(client, args.job_id)
    elif args.command == "files":
        list_files(client, purpose=None if args.all else "fine-tune")


if __name__ == "__main__":
    main()

