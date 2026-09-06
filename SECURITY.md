# Security model

LoRA Header Inspector handles untrusted headers and performs optional filesystem mutation. It is designed to fail closed, but it is not a substitute for backups or snapshots.

## Header parser

The parser:

- reads the 8-byte safetensors header length first;
- rejects implausible lengths before allocating the header buffer;
- requires valid UTF-8 and a JSON object root;
- rejects duplicate JSON keys;
- validates tensor names, dtypes, shapes, and offset ranges;
- normalizes scalar metadata and records warnings for malformed metadata;
- records size, device, inode, mtime, ctime, and raw-header hash for stale-plan checks;
- does not read tensor payloads during ordinary inspection.

`--hash-content` and duplicate comparison deliberately stream the complete file. They do not deserialize tensors.

## Path and symlink policy

By default, source symlinks and symlinked subdirectories are rejected. The library root itself may be deliberately selected through a symlink, but its resolved identity and parent topology are frozen in the plan.

`--allow-symlink-parents` is an explicit relaxation. Even then, topology is snapshotted and rechecked before mutation.

Destinations are constrained beneath `--base` unless `--allow-outside-base` is explicitly supplied. Portable path components are sanitized, and Windows reserved names are handled.

## No-overwrite publication

The transaction layer does not use an overwrite-capable rename as a final publication primitive.

- Same-filesystem moves first publish a hard link at a vacant destination and then remove the source.
- Cross-filesystem moves stream to a private temporary file, verify content, publish without overwrite, and only then remove the source.
- Hard-link replacement moves the old duplicate into a private backup before publishing the new link.

A destination that appears after preflight is treated as a race and is not overwritten.

## Journals

Apply and undo operations persist recoverable phases before mutation. Journals include source, destination, identity information, hashes where required, and per-step state. The first image of every apply, undo, or recovery journal is created with exclusive no-replace semantics before any model mutation starts.

Keep journals outside paths participating in the operation. The tool rejects a manifest path that overlaps a source, destination, sidecar, CSV, or protected journal path.

`--recover` performs a dry-run analysis by default. It will not remove or overwrite an unrelated file merely to force the filesystem back into the expected shape.

## Duplicate policy risks

### Hard links

Hard-linked paths share one inode. An in-place writer modifies all linked names. Treat model files as immutable and prefer update tools that write a new file and atomically replace one path.

### Quarantine

Quarantine changes old paths and may break saved workflows. Apply requires `--allow-workflow-path-breakage`.

### Sidecars

Exact model payloads can have different previews or metadata sidecars. Sidecars are compared and preserved independently. No sidecar is assumed identical merely because its model is identical.

## Network filesystems

NFS, SMB, FUSE, cloud mounts, and clustered filesystems may provide weaker durability, caching, hard-link, or `fsync` semantics. Test on a copy before applying to a production library. Hard-link policy may be unavailable or behave differently across mounts.

## Responsible reporting

Before publishing a security issue, provide the maintainer with:

- tool version;
- operating system and filesystem type;
- exact command;
- redacted plan and manifest fragments;
- whether the operation was a dry run, apply, undo, or recovery;
- a minimal reproducer that contains no model payload or private path.
