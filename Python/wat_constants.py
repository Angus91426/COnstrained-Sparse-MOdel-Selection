"""
Proportionality constants a1 and a2 of the WAT shape constraints (6)-(7) in the paper:

    beta_1 >= a1 * beta_3 + epsilon
    beta_2 >= a2 * beta_3 + epsilon

The values used in the paper come from the second author's domain expertise and are
withheld to protect trade secrets (footnote 2 of the paper). To run the WAT experiments,
supply your own values through the environment variables COSMOS_A1 and COSMOS_A2, e.g.

    export COSMOS_A1=1.2      # macOS / Linux
    export COSMOS_A2=1.0
    set COSMOS_A1=1.2         # Windows (cmd)

The explicit feasible point of Proposition 3 requires a2 >= 0 and a1 <= a2 + 1.
The real-data experiments (corn, riboflavin) do not use these constants.
"""
import os


def _read( name: str ) -> float:
    value = os.environ.get( name )
    if value is None:
        raise RuntimeError(
            f"Environment variable {name} is not set. The WAT proportionality constants used in "
            "the paper are withheld; set COSMOS_A1 and COSMOS_A2 to your own values "
            "(see Python/wat_constants.py)."
        )
    return float( value )


def a1() -> float:
    """Constant a1 in constraint (6): beta_1 >= a1 * beta_3 + epsilon."""
    return _read( 'COSMOS_A1' )


def a2() -> float:
    """Constant a2 in constraint (7): beta_2 >= a2 * beta_3 + epsilon."""
    return _read( 'COSMOS_A2' )
