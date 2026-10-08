import requests

from selectivity_pcm.paths import PROJECT_ROOT

PROTEINS = {
    "CDK1": "P06493",
    "CDK2": "P24941",
    "CDK4": "P11802",
    "CDK6": "Q00534",
}

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


def fetch_sequence(accession: str) -> str:
    url = UNIPROT_URL.format(accession=accession)

    response = requests.get(
        url,
        timeout=30,
    )
    response.raise_for_status()

    lines = response.text.strip().splitlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if not line.startswith(">")
    )

    if not sequence:
        raise ValueError(
            f"No sequence returned for {accession}"
        )

    return sequence


def main():
    fetched = {}

    for target, accession in PROTEINS.items():
        print(f"Fetching {target} ({accession})...")

        sequence = fetch_sequence(accession)

        print(
            f"  {len(sequence)} residues"
        )

        fetched[target] = {
            "uniprot": accession,
            "sequence": sequence,
        }

    OUTPUT_FILE = (
    PROJECT_ROOT
    / "examples"
    / "cdk"
    / "proteins.py"
    )   

    with OUTPUT_FILE.open("w",encoding="utf-8",) as handle:
        handle.write(
            '"""Canonical human protein sequences used by selectivity-pcm.\n\n'
        )
        handle.write(
            "Sequences retrieved from UniProtKB.\n"
        )
        handle.write(
            '"""\n\n'
        )

        handle.write("PROTEINS = {\n")

        for target, data in fetched.items():
            handle.write(
                f'    "{target}": {{\n'
            )
            handle.write(
                f'        "uniprot": "{data["uniprot"]}",\n'
            )
            handle.write(
                f'        "sequence": "{data["sequence"]}",\n'
            )
            handle.write(
                "    },\n"
            )

        handle.write("}\n")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()