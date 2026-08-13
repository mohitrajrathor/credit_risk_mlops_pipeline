"""Pytest configuration and custom hooks."""

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Pytest hook to generate a professional tests_report.md for CML output."""
    import os
    import time
    
    # Gather test outcomes
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    error = len(terminalreporter.stats.get('error', []))
    
    total = passed + failed + skipped + error
    duration = time.time() - terminalreporter._sessionstarttime
    
    status = "SUCCESS" if exitstatus == 0 else "FAILED"
    
    # Build GitHub Flavored Markdown report without icons for professional styling
    lines = [
        "# Credit Risk Pipeline - Test Execution Report",
        "",
        f"**Status**: {status}",
        "",
        "### Summary Metrics",
        f"- **Total Tests**: {total}",
        f"- **Passed**: {passed}",
        f"- **Failed**: {failed}",
        f"- **Skipped**: {skipped}",
        f"- **Errors**: {error}",
        f"- **Execution Time**: {duration:.2f} seconds",
        "",
        "### Test Case Statuses",
        "| Test Case | Status | Duration (s) |",
        "| :--- | :--- | :--- |"
    ]
    
    # Populate the table
    for rep in terminalreporter.stats.get('passed', []):
        lines.append(f"| `{rep.nodeid}` | passed | {rep.duration:.2f} |")
    for rep in terminalreporter.stats.get('failed', []):
        lines.append(f"| `{rep.nodeid}` | FAILED | {rep.duration:.2f} |")
    for rep in terminalreporter.stats.get('error', []):
        lines.append(f"| `{rep.nodeid}` | ERROR | {rep.duration:.2f} |")
        
    # Append traceback details for failures
    if failed or error:
        lines.append("")
        lines.append("### Failure Details",)
        for rep in (terminalreporter.stats.get('failed', []) + terminalreporter.stats.get('error', [])):
            lines.append("<details>")
            lines.append(f"<summary><b>{rep.nodeid}</b></summary>")
            lines.append("")
            lines.append("```python")
            lines.append(str(rep.longrepr))
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    # Write report file inside the reports directory
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "tests_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
