# Local project rules

## Buzz exception

TensorFold is Christian's solo project and has no Buzz project channel.
Christian explicitly authorized this exception on 2026-10-07.
Repository work here is exempt from Buzz lifecycle publication. All other
applicable engineering rules still apply.

## Upstream before patching

Christian requires patches to start from the latest upstream revision
(2026-10-10). Before editing a patch, fetch and verify the upstream branch.
Before building and again before promoting a binary, compare its pinned
upstream base with the live upstream branch tip. If it moved, rebase and retest.
Record the upstream commit, local patch commit, compiler version and binary
SHA256. Never call a build current using only an old release number.
Use `tools/local-m3/check_upstream.py` for this branch's freshness gate.
