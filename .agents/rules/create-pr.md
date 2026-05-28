---
trigger: manual
description: Triggered when the user requests the agent to create a Pull Request (PR) or submit changes to origin/GitHub.
---

# 🚀 Rule: Automated Pull Request Creation & Submission Workflow

## ⚡ Trigger Activation
This rule is **activated** whenever the user requests the agent to "create PR", "submit Pull Request", "open a PR", or push/stage changes for GitHub submission.

---

## 1. Pre-Submission Verification
Before committing or creating a Pull Request, the agent MUST run:
1. **Unit Verification**: Run `poetry run pytest` to ensure 100% green-passing tests across the entire suite (e.g., 291/291 passed).
2. **Badge Alignment**: Run `poetry run python main.py --sync` to synchronize package versioning and dynamic README badges.

---

## 2. Time-Tagged Branch Naming Standard
All new features, bug fixes, or review address streams must be committed to a dedicated feature branch with a distinct, time-tagged name following the convention:
`agy-YYMMDD-HHMM` (e.g., `agy-260528-0030`)
This branch must be checked out before committing:
`git checkout -b agy-YYMMDD-HHMM`

---

## 3. PR Description Drafting
The agent must draft a highly detailed, professional Markdown PR description detailing:
- Key context and problem summary.
- Comprehensive checklists of all changes/comments addressed.
- Verification and green-passing proof logs.
- Save this PR description as an artifact file in the conversation directory (e.g., `pr_review_refactor_description.md`) so the user has a durable copy.

---

## 4. Automated PR Creation via GitHub CLI (`gh`)
Once the branch is pushed (`git push origin <branch-name>`), use the GitHub CLI (`gh`) inline with the personal access token to automatically create the PR without interactive prompt blocks:
- **Token Location**: Retrieve the `GITHUB_TOKEN` or `GH_TOKEN` from `.env` or system environment variables.
- **PowerShell Invocation**:
  ```powershell
  $env:GH_TOKEN="<token>"; gh pr create --title "<title>" --body-file "<path-to-body-markdown-file>" --base main --head <branch-name>
  ```
- **Bash/Sh Invocation**:
  ```bash
  GH_TOKEN="<token>" gh pr create --title "<title>" --body-file "<path-to-body-markdown-file>" --base main --head <branch-name>
  ```