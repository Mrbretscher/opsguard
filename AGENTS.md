\# OpsGuard Repository Instructions



\## Project purpose



OpsGuard is a Python 3.11 predictive-maintenance portfolio project. It

acquires and validates equipment-failure data, trains baseline models,

evaluates their performance, and produces reproducible model artifacts and

reports.



The primary package is located in `src/opsguard/`.



\## Repository structure



\- `src/opsguard/`: application, data, modeling, training, evaluation, and

&#x20; plotting code

\- `tests/`: unit and integration tests

\- `docs/`: project documentation, evaluation plans, and data documentation

\- `scripts/`: setup, data-acquisition, training, and verification scripts

\- `data/`: local datasets and dataset documentation

\- `models/`: generated model artifacts

\- `reports/`: generated metrics and plots

\- `outputs/`: generated run outputs



\## Development environment



\- Assume Windows PowerShell unless the task explicitly specifies another shell.

\- Use Python 3.11.

\- Use the repository virtual environment at `.\\.venv`.

\- Invoke Python with `.\\.venv\\Scripts\\python.exe`.

\- Do not recreate or replace `.venv` unless explicitly requested.

\- Preserve the existing `src` package layout and editable installation.



\## Working rules



1\. Inspect the relevant code, documentation, and tests before editing.

2\. Check `git status --short` before making changes.

3\. Keep changes focused on the requested task.

4\. Do not perform unrelated refactoring.

5\. Preserve existing public interfaces unless a requested change requires

&#x20;  modifying them.

6\. Add or update tests whenever behavior changes.

7\. Update README or documentation when setup, commands, behavior, inputs,

&#x20;  outputs, or evaluation methodology changes.

8\. Do not commit, push, merge, or delete branches unless explicitly requested.

9\. Before declaring work complete, run the full verification script.

10\. Report the files changed, verification results, assumptions, and remaining

&#x20;   risks.



\## Required verification



Run the complete verification suite with:



```powershell

.\\scripts\\verify.ps1

