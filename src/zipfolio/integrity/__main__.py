"""``python -m zipfolio.integrity book.pkl [--mode ci] [--json report.json]``: the check from a shell or CI.

``book.pkl`` holds a dict of ``AccountingInputs`` fields (frames and series as pandas objects, the
opening state and contract specs as their dataclasses). Prints the status line and every check that is not
a pass; exits non-zero when the policy rejects the outcome (``IntegrityReport.exit_code``).
"""

import argparse
import pickle
import sys

from . import AccountingInputs, AccountingPolicy, check_integrity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zipfolio.integrity", description=__doc__.splitlines()[0])
    parser.add_argument("book", help="a pickle of AccountingInputs fields")
    parser.add_argument("--mode", choices=("report", "strict", "ci"), default="ci")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args(argv)
    with open(args.book, "rb") as handle:
        fields = pickle.load(handle)
    report = check_integrity(AccountingInputs(**fields), AccountingPolicy(mode=args.mode))
    if args.json:
        report.to_json(args.json)
    coverage = report.coverage
    print(f"accounting integrity: {report.status.value} (assurance {report.assurance_level.value}); "
          f"{coverage['passed']} passed, {coverage['failed']} failed, {coverage['warned']} warned, "
          f"{coverage['skipped']} skipped of {coverage['applicable']} applicable")
    for check in report.checks:
        if check.status.value not in ("PASS", "NOT_APPLICABLE"):
            print(f"  {check.status.value:<8} {check.id}: {check.explanation}")
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
