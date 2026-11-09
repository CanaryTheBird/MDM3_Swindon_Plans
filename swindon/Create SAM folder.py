from pathlib import Path
import pandas as pd

# Change this path to wherever your SAM is saved.
sam = pd.read_csv("swindon/swindon_sam.csv", index_col=0)

factors = ["LAB", "CAP"]
other_accounts = ["LAB", "CAP", "HOH", "GOV", "INV", "OUT"]
sectors = [account for account in sam.index
           if account not in other_accounts]

folder = Path("swindon/sam_input")
folder.mkdir(parents=True, exist_ok=True)

# Three lists identifying the account roles.
pd.DataFrame({"i": sectors}).to_csv(
    folder / "set-i-.csv", index=False
)
pd.DataFrame({"h": factors}).to_csv(
    folder / "set-h-.csv", index=False
)
pd.DataFrame({"u": sam.index}).to_csv(
    folder / "set-u-.csv", index=False
)

# Complete SAM, with its values unchanged.
sam.to_csv(folder / "param-sam-.csv", index_label="Account")