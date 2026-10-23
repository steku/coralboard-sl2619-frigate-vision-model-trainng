#!/usr/bin/env python3
"""
Report object class balance and annotation distribution across all CVAT tasks and jobs.

Connects to the CVAT REST API, inspects tasks and jobs, counts labeled annotations
(shapes, tracks, tags) per class, and prints a comprehensive balance report with
percentage breakdowns, imbalance ratios, and visual distribution bars.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

import config_utils

load_config = config_utils.load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and report object class balance across all CVAT tasks and jobs."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml containing CVAT credentials (default: auto-detected)",
    )
    parser.add_argument(
        "--cvat-url",
        default=None,
        help="CVAT server base URL (e.g. http://192.168.2.1:8085)",
    )
    parser.add_argument(
        "--cvat-user",
        default=None,
        help="CVAT username (or env CVAT_USER)",
    )
    parser.add_argument(
        "--cvat-pass",
        default=None,
        help="CVAT password (or env CVAT_PASS)",
    )
    parser.add_argument(
        "--task-id",
        nargs="*",
        default=["all"],
        help="Filter by specific CVAT Task ID(s) or 'all' (default: all)",
    )
    parser.add_argument(
        "--by-job",
        action="store_true",
        help="Include detailed per-job breakdown for each task",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON data instead of formatted tables",
    )
    return parser.parse_args()


def create_progress_bar(percentage: float, width: int = 24) -> str:
    """Create a unicode visual progress bar."""
    filled_len = int(round(width * (percentage / 100.0)))
    bar = "█" * filled_len + "░" * (width - filled_len)
    return bar


count_cvat_annotations = config_utils.count_cvat_annotations


def fetch_cvat_data(
    base_url: str,
    auth: Tuple[str, str],
    target_task_ids: List[str],
    include_jobs: bool = False,
) -> Dict[str, Any]:
    """Fetch tasks, jobs, and annotations from CVAT REST API."""
    session = requests.Session()
    session.auth = auth
    base_url = base_url.rstrip("/")

    # Verify server connectivity
    try:
        r = session.get(f"{base_url}/api/tasks?page_size=100", timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed connecting to CVAT at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks_data = r.json()
    all_tasks = tasks_data.get("results", [])

    # Filter task IDs if specified
    filter_all = any(str(t).lower() == "all" for t in target_task_ids)
    if not filter_all:
        wanted_ids = set()
        for t in target_task_ids:
            for piece in str(t).replace(",", " ").split():
                if piece.isdigit():
                    wanted_ids.add(int(piece))
        selected_tasks = [t for t in all_tasks if t["id"] in wanted_ids]
    else:
        selected_tasks = all_tasks

    selected_tasks.sort(key=lambda t: t["id"])

    task_reports = []
    overall_class_counts = Counter()
    total_frames = 0
    total_annotations = 0

    for task in selected_tasks:
        tid = task["id"]
        tname = task.get("name", f"Task {tid}")
        tmode = task.get("mode", "annotation")
        tstatus = task.get("status", "unknown")
        tsize = task.get("size", 0)
        total_frames += tsize

        # Retrieve label mappings for this task
        r_labels = session.get(f"{base_url}/api/labels?task_id={tid}&page_size=100", timeout=10).json()
        label_id_to_name = {
            lbl["id"]: lbl["name"]
            for lbl in r_labels.get("results", [])
        }

        # Retrieve task-level annotations
        r_ann = session.get(f"{base_url}/api/tasks/{tid}/annotations", timeout=15).json()
        task_counts = count_cvat_annotations(r_ann, label_id_to_name)

        task_total_ann = sum(task_counts.values())
        total_annotations += task_total_ann
        overall_class_counts.update(task_counts)

        # Retrieve jobs if requested
        jobs_list = []
        if include_jobs:
            r_jobs = session.get(f"{base_url}/api/jobs?task_id={tid}&page_size=100", timeout=10).json()
            for j in r_jobs.get("results", []):
                jid = j["id"]
                jstate = j.get("state", "unknown")
                jstage = j.get("stage", "unknown")
                jstatus = j.get("status", "unknown")

                r_jann = session.get(f"{base_url}/api/jobs/{jid}/annotations", timeout=10).json()
                job_counts = count_cvat_annotations(r_jann, label_id_to_name)

                jobs_list.append({
                    "id": jid,
                    "state": jstate,
                    "stage": jstage,
                    "status": jstatus,
                    "annotation_count": sum(job_counts.values()),
                    "class_counts": dict(job_counts),
                })

        task_reports.append({
            "id": tid,
            "name": tname,
            "status": tstatus,
            "mode": tmode,
            "frames": tsize,
            "labels": list(label_id_to_name.values()),
            "annotation_count": task_total_ann,
            "class_counts": dict(task_counts),
            "jobs": jobs_list,
        })

    return {
        "tasks": task_reports,
        "overall_class_counts": dict(overall_class_counts),
        "total_tasks": len(selected_tasks),
        "total_frames": total_frames,
        "total_annotations": total_annotations,
    }


def print_text_report(data: Dict[str, Any], include_jobs: bool = False):
    """Print beautifully formatted ASCII report."""
    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    BOLD = "\033[1m"
    NC = "\033[0m"

    tasks = data["tasks"]
    overall_counts = data["overall_class_counts"]
    total_ann = data["total_annotations"]
    total_frames = data["total_frames"]

    print("")
    print(f"{BOLD}========================================================================================{NC}")
    print(f"{CYAN}{BOLD}                    CVAT DATASET OBJECT BALANCE AUDIT REPORT{NC}")
    print(f"{BOLD}========================================================================================{NC}")
    print(f" Total Tasks Scanned : {data['total_tasks']}")
    print(f" Total Image Frames  : {total_frames:,}")
    print(f" Total Annotations   : {total_ann:,}")
    print(f"{BOLD}========================================================================================{NC}")

    # 1. Per-Task Breakdown
    print(f"\n{BOLD}----------------------------------------------------------------------------------------{NC}")
    print(f"{BOLD}  PER-TASK OBJECT BREAKDOWN{NC}")
    print(f"{BOLD}----------------------------------------------------------------------------------------{NC}")

    for t in tasks:
        tid = t["id"]
        tname = t["name"]
        tframes = t["frames"]
        tann = t["annotation_count"]
        status_color = GREEN if t["status"] in ("completed", "annotation") else YELLOW

        print(f"\n{BOLD}Task #{tid}:{NC} {tname}")
        print(f"  Status: {status_color}{t['status']}{NC} | Frames: {tframes} | Annotations: {tann}")

        if not t["class_counts"]:
            print(f"  {YELLOW}(No annotations found in this task){NC}")
        else:
            sorted_classes = sorted(t["class_counts"].items(), key=lambda x: x[1], reverse=True)
            items_str = []
            for cname, ccount in sorted_classes:
                pct = (ccount / tann * 100.0) if tann > 0 else 0.0
                items_str.append(f"{cname}: {BOLD}{ccount}{NC} ({pct:.1f}%)")
            print(f"  Objects: {' | '.join(items_str)}")

        if include_jobs and t["jobs"]:
            print(f"  Jobs ({len(t['jobs'])}):")
            for j in t["jobs"]:
                jid = j["id"]
                jstate = j["state"]
                jann = j["annotation_count"]
                j_counts_str = ", ".join([f"{k}: {v}" for k, v in j["class_counts"].items()]) or "none"
                print(f"    - Job #{jid} [{jstate}]: {jann} annotations ({j_counts_str})")

    # 2. Overall Class Balance & Imbalance Ratios
    print(f"\n{BOLD}========================================================================================{NC}")
    print(f"{BOLD}  OVERALL CLASS DISTRIBUTION & BALANCE RATIOS{NC}")
    print(f"{BOLD}========================================================================================{NC}")
    print(f" {'Class':<12} {'Count':>7}   {'Share':>7}   {'Distribution':<24}   {'Ratio vs Max':>14}   {'Health':<10}")
    print(f" {'-'*12} {'-'*7}   {'-'*7}   {'-'*24}   {'-'*14}   {'-'*10}")

    if not overall_counts:
        print(f" {RED}No annotations found across any tasks!{NC}")
        print(f"{BOLD}========================================================================================{NC}\n")
        return

    sorted_overall = sorted(overall_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_overall[0][1] if sorted_overall else 1

    for cname, count in sorted_overall:
        pct = (count / total_ann * 100.0) if total_ann > 0 else 0.0
        bar = create_progress_bar(pct, width=24)
        ratio_str = f"1 : {max_count / count:.1f}x" if count > 0 else "N/A"

        # Determine health status based on balance ratio and absolute count
        ratio = max_count / count if count > 0 else 999.0
        if count < 20 or ratio > 5.0:
            health = f"{RED}CRITICAL{NC}"
        elif count < 100 or ratio > 2.5:
            health = f"{YELLOW}LOW{NC}"
        else:
            health = f"{GREEN}GOOD{NC}"

        print(f" {cname:<12} {count:>7}   {pct:>6.1f}%   {CYAN}{bar}{NC}   {ratio_str:>14}   {health}")

    print(f"{BOLD}========================================================================================{NC}")

    # 3. Actionable Insights & Recommendations
    print(f"\n{BOLD}Key Observations & Imbalance Diagnosis:{NC}")
    top_class, top_count = sorted_overall[0]
    underrepresented = [c for c, cnt in sorted_overall if (max_count / cnt if cnt > 0 else 999.0) > 2.5 or cnt < 100]

    if underrepresented:
        print(f" • Dominant Class : {BOLD}{top_class}{NC} ({top_count:,} instances, {top_count / total_ann * 100:.1f}% of dataset)")
        print(f" • Underrepresented: {YELLOW}{', '.join(underrepresented)}{NC}")
        for c in underrepresented:
            cnt = overall_counts[c]
            r = max_count / cnt if cnt > 0 else 0
            if c == "dog":
                print(f"   - {BOLD}dog{NC} has only {cnt} instances vs {top_count} {top_class} ({r:.1f}x deficit).")
                print(f"     This directly causes false classifications (e.g., dogs being mispredicted as horses or cars).")
            elif c == "cat":
                print(f"   - {BOLD}cat{NC} has only {cnt} instances. Model will almost never trigger detections on cats.")
            elif c == "face":
                print(f"   - {BOLD}face{NC} has only {cnt} instances. Insufficient for reliable detection.")

        print(f"\n{BOLD}Recommended Remediation:{NC}")
        print(f" 1. Enable CVAT-balanced harvesting in config.yaml: {CYAN}frigate.balance_mode: \"cvat-deficit\"{NC} (or run with {CYAN}--balance-mode cvat-deficit{NC}).")
        print(f" 2. Harvest focused batches for: {YELLOW}{', '.join(underrepresented)}{NC} before retraining.")
        print(f" 3. Make sure to export ALL tasks via: {CYAN}python3 export-cvat-to-yolo.py --export all{NC}")
    else:
        print(f" • Dataset classes are well-balanced across all categories.")

    print("")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cvat_url, cvat_user, cvat_pass = config_utils.get_cvat_credentials(
        cfg, url=args.cvat_url, user=args.cvat_user, password=args.cvat_pass
    )

    if not cvat_user or not cvat_pass:
        print("[ERROR] CVAT username and password required in config.yaml or via --cvat-user / --cvat-pass.", file=sys.stderr)
        sys.exit(1)

    auth = (cvat_user, cvat_pass)
    data = fetch_cvat_data(
        base_url=cvat_url,
        auth=auth,
        target_task_ids=args.task_id,
        include_jobs=args.by_job,
    )

    if args.output_json:
        print(json.dumps(data, indent=2))
    else:
        print_text_report(data, include_jobs=args.by_job)


if __name__ == "__main__":
    main()
