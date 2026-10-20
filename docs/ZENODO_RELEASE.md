# Zenodo handoff

The technical artifact plan is prepared locally; **no Zenodo record, DOI or GitHub release has been created**. The author metadata is transcribed from the current manuscript under the user's instruction. Original-code rights remain unchanged.

## Planned deposit

The prepared data/model plan has **99 files / approximately 43.99 GB**. Add the committed GitHub software source ZIP as file 100. This stays below Zenodo's default [100-file / 50 GB record limits](https://help.zenodo.org/docs/deposit/create-new-upload/). A ZIP-only upload of all 42 GB of weights is unnecessary: 96 large checkpoint files are uploaded directly from their authenticated original paths; 36 smaller CNN checkpoint files share one ZIP. The other two ZIPs contain study inputs/records and artifact metadata.

`configs/zenodo_artifacts.json` is the public 99-file manifest, without local PC paths. `configs/checkpoint_artifacts.json` maps every one of the 132 fitted weights; input and preparation manifests cover raw sources, pretrained snapshots, split files and fit records. The final software ZIP is identified in the private handoff plan after commit, avoiding a self-referential hash inside its own source tree.

## Metadata and rights

Use `CITATION.cff`, `docs/CITATION_PROVENANCE.json` and `docs/ZENODO_METADATA_DRAFT.json`. Creator order is Arian Rahman Aditta, Udita Rahman Arshee, Shafin Ahmed, Sheikh Md. Rabiul Islam and Nishat Taaha. Only the ORCID actually supplied in the manuscript is included. Author emails and manuscript Word files are excluded.

Zenodo defaults to CC BY 4.0; [its license instructions](https://help.zenodo.org/docs/deposit/describe-records/licenses/) also support custom licenses. Select and inspect a **custom/mixed rights notice** matching the root LICENSE and upstream notices. Do not apply CC BY or another permissive license to original code on the authors' behalf. Existing third-party licenses and source-data terms remain applicable. The draft JSON is a human-review metadata source, not an automatic API submission assigning a license identifier.

No `.zenodo.json` is supplied that could override citation data or silently choose release licensing. For this combined software/data/model record, use a manually reviewed draft. If a GitHub-linked software-only record is chosen instead, its DOI does not by itself make the separately required model/data artifacts available.

## Upload sequence

1. Confirm the committed GitHub source and all files in the local upload plan are final. Preserve the original research directory.
2. Create an editable draft in Zenodo and review title, creators, description, custom rights and the repository URL. Use the actual draft ID, never a placeholder DOI in the manuscript.
3. In a private shell, provide a Zenodo deposit-write token as `ZENODO_ACCESS_TOKEN`; never put it in source code, notebooks, logs or chat. Run `scripts/upload_to_zenodo.py --draft-id ACTUAL_ID --local-plan PATH_TO_PRIVATE_PLAN` to inspect the plan, then add `--execute` to upload. The command authenticates each local SHA-256, streams each file, checks Zenodo's MD5, reuses matching uploaded files and refuses conflicting files. It does not create a record, change metadata or publish.
4. Inspect the draft files and metadata in Zenodo. Publication is a separate explicit action. After publication, add the real record ID/DOI to repository metadata and the manuscript, test `scripts/download_artifacts.py` against the live record, and commit that small update. Retain the deposited software snapshot's commit identity.

The upload helper has offline integrity/transport tests and a local acquisition test. Live Zenodo API transfer and public-record download cannot be claimed before an actual draft/deposit exists. Official [API documentation](https://developers.zenodo.org/) describes bucket streaming and returned file checksums.
