# output/

The two output files are not stored in git because of GitHub's 100 MB file limit
(`candidate_pairs.tsv` is 137 MB). They are in the submission zip, `Ctrl_Alt_Del_submission.zip`:

| File | Size | md5 |
|---|---|---|
| `matching_results.tsv` | 98.1 MB | `fd68d976f8f08a50f4c00be267a37ce9` |
| `candidate_pairs.tsv` | 137.1 MB | `18d9c5abe12ca182c3afc151e0cebd85` |

`matching_results.tsv` is the file uploaded to the leaderboard (public score 0.987462).
`candidate_pairs.tsv` has 8,906,044 pairs, 5.14 candidates per Source 1 entity.
`code/business_entity_resolution/run_final.sh` regenerates both.
