from opencadd.databases.klifs import setup_remote

TARGETS = {
    "ROCK2": "O75116",
    "ROCK1": "Q13464",
    "CSNK2A2": "P19784",
    "AURKA": "O14965",
    "STK3": "Q13188",
}


def main():

    session = setup_remote()

    print(
        "Checking KLIFS kinase coverage..."
    )

    for target_name, uniprot_id in TARGETS.items():

        print()
        print(
            f"=== {target_name} ({uniprot_id}) ==="
        )

        kinases = session.kinases.all_kinases()

        matches = kinases[
            kinases[
                "kinase.uniprot"
            ]
            == uniprot_id
        ]

        if matches.empty:
            print(
                "No KLIFS kinase entry found."
            )
            continue

        print(
            f"KLIFS kinase entries: "
            f"{len(matches)}"
        )

        print(
            matches[
                [
                    "kinase.klifs_id",
                    "kinase.klifs_name",
                    "kinase.full_name",
                    "kinase.gene_name",
                    "kinase.uniprot",
                ]
            ]
            .to_string(
                index=False
            )
        )

        for _, row in matches.iterrows():

            klifs_id = int(
                row[
                    "kinase.klifs_id"
                ]
            )

            structures = (
                session.structures
                .by_kinase_klifs_id(
                    klifs_id
                )
            )

            print(
                f"Structures for KLIFS ID "
                f"{klifs_id}: "
                f"{len(structures)}"
            )

            if not structures.empty:

                columns = [
                    column
                    for column in [
                        "structure.klifs_id",
                        "structure.pdb_id",
                        "structure.resolution",
                        "structure.pocket",
                    ]
                    if column
                    in structures.columns
                ]

                print(
                    structures[
                        columns
                    ]
                    .head(10)
                    .to_string(
                        index=False
                    )
                )


if __name__ == "__main__":
    main()