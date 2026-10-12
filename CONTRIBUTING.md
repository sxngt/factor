# Contributing / 기여

Thanks for your interest. Issues and pull requests are welcome.

## Setup / 준비

```
uv sync --extra dev --extra real
uv run pytest -q
```

Isaac Lab, ROS 2 and the FAIRINO SDK are only needed for the twin, the driver and the robot
(see the README); the unit tests run without them.

## Before a pull request / PR 전에

```
uv run ruff format src tests
uv run ruff check src tests
uv run pytest -q
```

CI runs the same on Python 3.10–3.12.

## Style / 스타일

* Docstrings and comments in English first, then Korean (after ` / ` or on the next line), both
  carrying the same meaning.
* Readable multi-line code over dense one-liners; line length 110 (ruff).
* Comments explain why, not history; dates and measurements belong in `docs/`.
* Units: metres and radians; degrees only where a name says `_deg`.
* New behaviour that can be tested without a GPU, ROS or a robot gets a unit test (factor the
  logic into a pure function if needed, as `factor.robots.fr3.servo.servo_step`).

## Safety / 안전

Changes to `factor.robots.fr3` (driver, servo limiter, safety monitor) or `factor.real` that
affect motion must be tried with `--fake` first, then on the robot at reduced speed with the
e-stop in reach. Describe how you tested them in the pull request.
