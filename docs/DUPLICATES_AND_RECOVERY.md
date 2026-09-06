# Duplicates, transactions, undo, and recovery

## Collision classification

### `same_inode`

Several paths already refer to the same filesystem object. They are aliases, not independent copies.

### `exact_duplicate`

Several independent regular files have the same size and full SHA-256. Different inodes confirm that they consume separate logical file entries.

### `existing_exact_duplicate`

The normalized destination already exists and has the same content as the planned source. The existing destination is normally the canonical object.

### `different_content`

The target filename is shared, but full hashes differ. The group is blocked. Automatic renaming or deduplication would risk changing model identity.

### `unverified_collision`

The group could not be safely hashed or identified. It is blocked.

### `sidecar_collision`

The model move may be clear, but one or more preview or metadata sidecars converge on the same destination. Sidecars are separate evidence and cannot be overwritten casually.

## Canonical selection

The planner uses deterministic priority:

1. an existing normalized destination;
2. a unique reviewer `canonical=True` choice;
3. a unique `duplicate_preference=canonical` choice;
4. richer recognized sidecar coverage;
5. shorter path;
6. lexical order.

Conflicting reviewer choices block the group.

## Report policy

`report` records what was found and changes nothing. Use it to audit a new library or validate classification changes.

## Hard-link policy

For an exact duplicate group, the engine keeps public paths but gives them one inode.

The operation is backup-first:

1. move the independent duplicate to a private backup path;
2. publish a hard link at the now-vacant public path without overwrite;
3. verify identity;
4. remove the private backup only after success.

If another process occupies the public path, the intruder is preserved and the original duplicate remains recoverable from its private backup.

Undo breaks the hard link back into an independent copy. It does not assume that the path still shares the expected inode without checking.

## Quarantine policy

The canonical file enters the normalized directory. Redundant copies and their own sidecars enter a hash-scoped quarantine tree. No file is deleted. Undo restores original paths when they remain vacant and identities match.

## Cross-filesystem moves

A rename cannot cross devices. The fallback therefore:

1. streams the source into a private destination-side temporary file;
2. calculates and verifies the content hash;
3. publishes without overwrite;
4. records the published destination identity;
5. removes the source only after publication succeeds.

A failure before source removal leaves the source intact. Recovery can remove a destination that the journal proves belongs to the interrupted operation, but it will not remove an unrelated replacement.

## Apply journals

The manifest is created before the first mutation. Each operation and step is added as pending before it begins. Private paths, identities, hashes, and phase transitions make an interrupted state analyzable.

## Undo journals

Undo first preflights every operation. It then journals each reverse step. If part of an operation fails, completed reverse steps are rolled back to the applied state where possible.

## Recovery

Always dry-run first:

```bash
python3 inspect_sort_lora.py --recover manifest.json
```

Recovery determines whether the journal describes an apply or undo operation and inspects actual paths. It may conclude that automatic recovery is blocked because contents or identities no longer match. That is preferable to forcing a neat but false history onto the filesystem.
