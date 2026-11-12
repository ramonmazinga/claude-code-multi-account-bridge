"""Hostile sequences against light backups and log trimming.

Every scenario runs the REAL functions (or the real command line through
bridge.main) on a synthetic tree under a temp dir. Where a race has to be
staged, the ENVIRONMENT is changed at a known moment by wrapping shutil.copy2
(only backup_changes calls it directly), never the code under test.
"""

import contextlib
import datetime
import io
import os
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

import bridge_testlib as tl
import bridge

UTC = datetime.timezone.utc
STAMP = "%Y%m%dT%H%M%S%fZ"
KB = 1024
MB = 1024 * 1024


def chmod_if_present(path, mode):
    if os.path.lexists(path) and not os.path.islink(path):
        os.chmod(path, mode)


def tree_bytes(path):
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            total += os.lstat(os.path.join(current, name)).st_size
    return total


def files_of(path):
    """relpath -> (bytes, mtime_ns) for every regular file under path."""
    return {rel: (payload, mtime) for rel, (kind, payload, mtime)
            in tl.snapshot(path).items() if kind == "file"}


class HostileCase(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="bridge-test-")
        self.addCleanup(shutil.rmtree, self.base, True)
        self.user_data = os.path.join(self.base, "userData")
        self.state_dir = os.path.join(self.base, "state")
        self.backups_root = os.path.join(self.state_dir, "backups")
        self.root = tl.make_root(self.user_data)
        self.a = tl.make_leaf(self.user_data, tl.uid(1), tl.uid(11))
        self.b = tl.make_leaf(self.user_data, tl.uid(2), tl.uid(22))
        self.c = tl.make_leaf(self.user_data, tl.uid(3), tl.uid(33))
        self.leaves = [self.a, self.b, self.c]
        # NEVER the real ~/Library/Logs.
        self.home = os.path.join(self.base, "home")
        patcher = mock.patch.dict(os.environ, {"HOME": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)

    # ------------------------------------------------------------ helpers
    def now(self):
        return datetime.datetime.now(UTC)

    def full_backup(self, when, with_tree=True):
        path = os.path.join(self.backups_root, when.strftime(STAMP))
        os.makedirs(os.path.join(path, tl.SESSIONS_DIRNAME) if with_tree
                    else path)
        return path

    def recent_full_backup(self):
        return self.full_backup(self.now() - datetime.timedelta(hours=1))

    def names(self):
        if not os.path.isdir(self.backups_root):
            return []
        return sorted(os.listdir(self.backups_root))

    def changes_folders(self):
        return [n for n in self.names() if n.endswith("-changes")]

    def full_folders(self):
        return [n for n in self.names() if bridge.STAMP_RE.match(n)]

    def sync(self):
        return bridge.sync_once(self.root, execute=True,
                                state_dir=self.state_dir)

    def run_cli(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bridge.main(["sync", "--user-data", self.user_data,
                                "--state-dir", self.state_dir] + list(extra),
                               runner=tl.RecordingRunner())
        return code, out.getvalue()

    def read(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def overwrite_pair(self, sid):
        """A newer copy in a, a stale copy in b: one overwrite of b."""
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        return tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                               title="old")

    def saved(self, folder, original):
        return os.path.join(self.backups_root, folder, tl.SESSIONS_DIRNAME,
                            os.path.relpath(original, self.root))

    def wrap_copy2(self, before=None, after=None):
        """Pass-through shutil.copy2 that changes the world around each call."""
        real = shutil.copy2
        calls = []

        def wrapper(src, dst, *args, **kwargs):
            calls.append(src)
            if before:
                before(src, len(calls))
            result = real(src, dst, *args, **kwargs)
            if after:
                after(src, len(calls))
            return result

        patcher = mock.patch.object(shutil, "copy2", wrapper)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls


# ===================================================================== backups

class SixtyRunsTest(HostileCase):
    IDS = 200
    PAD = 50 * KB

    def record(self, n, version):
        return tl.record_bytes(tl.uid(1000 + n), last_activity=1000,
                               title="v%d" % version,
                               extra={"pad": "p" * self.PAD})

    def test_sixty_writing_runs_stay_small_and_save_every_overwrite(self):
        mtime = 1_000_000_000_000
        for n in range(self.IDS):
            for leaf in self.leaves:
                tl.write_record(leaf, tl.uid(1000 + n), mtime_ns=mtime,
                                raw=self.record(n, 0))
        self.assertTrue(bridge.plan_sync(bridge.gather(self.root)).is_empty())

        for run in range(60):
            n = (run * 7) % self.IDS
            sid = tl.uid(1000 + n)
            winner_leaf = self.leaves[run % 3]
            losers = [os.path.join(leaf, "local_%s.json" % sid)
                      for leaf in self.leaves if leaf != winner_leaf]
            old = {path: (self.read(path), os.lstat(path).st_mtime_ns)
                   for path in losers}
            mtime += 1_000_000_000
            tl.write_record(winner_leaf, sid, mtime_ns=mtime,
                            raw=self.record(n, run + 1))
            before = set(self.names())

            result = self.sync()

            self.assertEqual(result.applied["errors"], 0, run)
            self.assertEqual(result.applied["copies_written"], 2, run)
            new = sorted(set(self.names()) - before)
            self.assertEqual(len(new), 1, (run, new))
            if run == 0:
                self.assertRegex(new[0], bridge.STAMP_RE)
            else:
                self.assertTrue(new[0].endswith("-changes"), (run, new))
            self.assertEqual(result.backup,
                             os.path.join(self.backups_root, new[0]))
            for path, (data, old_mtime) in old.items():
                copy = self.saved(new[0], path)
                self.assertEqual(self.read(copy), data, (run, path))
                self.assertEqual(os.lstat(copy).st_mtime_ns, old_mtime,
                                 (run, path))
            if run:
                self.assertEqual(len(files_of(result.backup)), 2, run)

        tree = tree_bytes(self.root)
        backups = tree_bytes(self.backups_root)
        sys.stderr.write(
            "\n[60 runs] tree=%d bytes backups=%d bytes fulls=%d "
            "changes=%d ratio=%.3f\n" % (
                tree, backups, len(self.full_folders()),
                len(self.changes_folders()), backups / float(tree)))
        self.assertEqual(len(self.full_folders()), 1)
        self.assertEqual(len(self.changes_folders()), 59)
        self.assertLess(backups, 2 * tree)


class DeleteAndAddTest(HostileCase):
    def test_a_tombstone_run_saves_every_deleted_records_bytes(self):
        sid = tl.uid(5)
        tl.write_tombstone(self.a, sid, mtime_ns=8_000_000_000)
        doomed = [tl.write_record(self.b, sid, mtime_ns=2_000_000_000,
                                  title="b"),
                  tl.write_record(self.c, sid, mtime_ns=3_000_000_000,
                                  title="c")]
        old = {path: (self.read(path), os.lstat(path).st_mtime_ns)
               for path in doomed}
        self.recent_full_backup()

        code, _ = self.run_cli("--execute")

        self.assertEqual(code, 0)
        changes = self.changes_folders()
        self.assertEqual(len(changes), 1)
        for path, (data, mtime) in old.items():
            self.assertFalse(os.path.exists(path))
            copy = self.saved(changes[0], path)
            self.assertEqual(self.read(copy), data)
            self.assertEqual(os.lstat(copy).st_mtime_ns, mtime)
        self.assertEqual(len(files_of(os.path.join(self.backups_root,
                                                   changes[0]))), 2)

    def test_a_run_that_only_adds_makes_no_folder(self):
        for n in range(5):
            tl.write_record(self.leaves[n % 3], tl.uid(50 + n),
                            mtime_ns=5_000_000_000 + n)
        recent = self.recent_full_backup()

        code, out = self.run_cli("--execute")

        self.assertEqual(code, 0)
        self.assertEqual(self.names(), [os.path.basename(recent)])
        self.assertTrue(bridge.plan_sync(bridge.gather(self.root)).is_empty())

    def test_a_second_run_right_after_a_writing_run_makes_nothing(self):
        self.overwrite_pair(tl.uid(5))
        self.recent_full_backup()
        self.assertEqual(self.run_cli("--execute")[0], 0)
        after_first = tl.snapshot(self.state_dir)
        tree = tl.snapshot(self.root)

        code, _ = self.run_cli("--execute")

        self.assertEqual(code, 0)
        self.assertEqual(tl.snapshot(self.state_dir), after_first)
        self.assertEqual(tl.snapshot(self.root), tree)

    def test_a_second_run_after_a_first_ever_full_backup_makes_nothing(self):
        self.overwrite_pair(tl.uid(5))
        self.assertEqual(self.run_cli("--execute")[0], 0)
        self.assertEqual(len(self.full_folders()), 1)
        names = self.names()

        self.assertEqual(self.run_cli("--execute")[0], 0)

        self.assertEqual(self.names(), names)


class RaceTest(HostileCase):
    """The desktop app rewrites a destination while the run is under way."""

    APP_BYTES = b'{"id": "app wrote this"}'

    def app_rewrites(self, path, mtime_ns=20_000_000_000):
        with open(path, "wb") as handle:
            handle.write(self.APP_BYTES)
        tl.set_mtime_ns(path, mtime_ns)

    def test_rewrite_after_the_backup_copied_it_is_skipped_not_lost(self):
        dest = self.overwrite_pair(tl.uid(5))
        old = (self.read(dest), os.lstat(dest).st_mtime_ns)
        self.recent_full_backup()
        calls = self.wrap_copy2(after=lambda src, n: self.app_rewrites(dest))

        code, out = self.run_cli("--execute")

        self.assertEqual(len(calls), 1)
        self.assertEqual(code, 0)
        self.assertIn("skipped=1", out)
        self.assertIn("errors=0", out)
        self.assertEqual(self.read(dest), self.APP_BYTES)
        self.assertEqual(os.lstat(dest).st_mtime_ns, 20_000_000_000)
        copy = self.saved(self.changes_folders()[0], dest)
        self.assertEqual((self.read(copy), os.lstat(copy).st_mtime_ns), old)

    def test_rewrite_between_plan_and_backup_is_skipped_not_lost(self):
        dest = self.overwrite_pair(tl.uid(5))
        self.recent_full_backup()
        calls = self.wrap_copy2(before=lambda src, n: self.app_rewrites(dest))

        code, out = self.run_cli("--execute")

        self.assertEqual(len(calls), 1)
        self.assertEqual(code, 0)
        self.assertIn("skipped=1", out)
        self.assertEqual(self.read(dest), self.APP_BYTES)
        copy = self.saved(self.changes_folders()[0], dest)
        self.assertEqual(self.read(copy), self.APP_BYTES)

    def test_a_destination_deleted_before_the_backup_is_skipped(self):
        first = self.overwrite_pair(tl.uid(5))
        tl.write_record(self.a, tl.uid(6), mtime_ns=9_000_000_000, title="n")
        second = tl.write_record(self.b, tl.uid(6), mtime_ns=1_000_000_000,
                                 title="o")
        self.recent_full_backup()

        def vanish(src, n):
            other = second if src == first else first
            if n == 1:
                os.remove(other)
        self.wrap_copy2(before=vanish)

        code, out = self.run_cli("--execute")

        self.assertEqual(code, 0)
        self.assertIn("errors=0", out)


class HalfwayFailureTest(HostileCase):
    def test_a_source_turning_unreadable_halfway_applies_nothing(self):
        first = self.overwrite_pair(tl.uid(5))
        tl.write_record(self.c, tl.uid(6), mtime_ns=9_000_000_000, title="n")
        second = tl.write_record(self.a, tl.uid(6), mtime_ns=1_000_000_000,
                                 title="o")
        self.recent_full_backup()
        os.makedirs(os.path.join(self.backups_root,
                                 "20200101T000000000000Z-changes", "x"))
        tl.write_other(os.path.join(self.backups_root,
                                    "20200101T000000000000Z-changes", "x"),
                       "f", b"earlier")
        earlier = tl.snapshot(self.backups_root)
        tree = tl.snapshot(self.root)
        for path in (first, second):
            self.addCleanup(chmod_if_present, path, 0o600)

        def lock_the_other(src, n):
            if n == 1:
                os.chmod(second if src == first else first, 0o000)
        calls = self.wrap_copy2(after=lock_the_other)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            code, out = self.run_cli("--execute")

        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        for path in (first, second):
            os.chmod(path, 0o600)
        self.assertEqual(tl.snapshot(self.root), tree)
        self.assertEqual(tl.snapshot(self.backups_root), earlier)


class BackupsFolderShapeTest(HostileCase):
    def test_a_symlinked_backups_folder_is_used_and_left_a_symlink(self):
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        os.makedirs(self.state_dir)
        os.symlink(elsewhere, self.backups_root)
        self.overwrite_pair(tl.uid(5))

        self.assertEqual(self.run_cli("--execute")[0], 0)
        self.overwrite_pair(tl.uid(6))
        self.assertEqual(self.run_cli("--execute")[0], 0)

        self.assertTrue(os.path.islink(self.backups_root))
        inside = sorted(os.listdir(elsewhere))
        self.assertEqual(len(inside), 2)
        self.assertRegex(inside[0], bridge.STAMP_RE)
        self.assertTrue(inside[1].endswith("-changes"))

    def test_a_regular_file_as_backups_folder_applies_nothing(self):
        os.makedirs(self.state_dir)
        tl.write_other(self.state_dir, "backups", b"not a folder")
        self.overwrite_pair(tl.uid(5))
        tree = tl.snapshot(self.root)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            code, _ = self.run_cli("--execute")

        self.assertEqual(code, 1)
        self.assertEqual(tl.snapshot(self.root), tree)
        self.assertEqual(self.read(self.backups_root), b"not a folder")

    def test_a_read_only_empty_backups_folder_applies_nothing(self):
        os.makedirs(self.backups_root)
        os.chmod(self.backups_root, 0o500)
        self.addCleanup(chmod_if_present, self.backups_root, 0o700)
        self.overwrite_pair(tl.uid(5))
        tree = tl.snapshot(self.root)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            code, _ = self.run_cli("--execute")

        self.assertEqual(code, 1)
        self.assertEqual(tl.snapshot(self.root), tree)
        os.chmod(self.backups_root, 0o700)
        self.assertEqual(self.names(), [])

    def test_a_changes_named_regular_file_is_never_touched(self):
        self.recent_full_backup()
        junk = tl.write_other(self.backups_root,
                              "20200101T000000000000Z-changes", b"junk")
        original = bridge.CHANGES_KEPT
        bridge.CHANGES_KEPT = 1
        self.addCleanup(setattr, bridge, "CHANGES_KEPT", original)
        self.overwrite_pair(tl.uid(5))

        self.assertEqual(self.run_cli("--execute")[0], 0)

        self.assertEqual(self.read(junk), b"junk")
        self.assertEqual(len([n for n in self.changes_folders()
                              if os.path.isdir(os.path.join(
                                  self.backups_root, n))]), 1)

    def test_names_that_only_look_like_backups_never_count_as_full(self):
        now = self.now()
        recent = (now - datetime.timedelta(hours=1)).strftime(STAMP)
        for name in (recent + "-changes-changes", recent + "x",
                     "20261399T000000000000Z", "2026111T000000000000Z"):
            os.makedirs(os.path.join(self.backups_root, name,
                                     tl.SESSIONS_DIRNAME))

        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=now))

    def test_prune_never_removes_or_counts_a_folder_not_named_a_stamp(self):
        """needs_full_backup says only a stamp-named folder is a backup."""
        fulls = ["20200101T0000%02d000000Z" % i for i in range(12)]
        junk = ["00-notes", "20200101T000000000000Zx",
                "20201399T000000000000Z", "99-zz-notes"]
        os.makedirs(self.backups_root)
        for name in fulls + junk:
            os.makedirs(os.path.join(self.backups_root, name))

        bridge.prune_backups(self.backups_root)

        left = self.names()
        for name in junk:
            self.assertIn(name, left)
        self.assertEqual([n for n in fulls if n in left], fulls[-10:])

    def test_1005_changes_folders_are_cut_to_the_cap_newest_kept(self):
        os.makedirs(self.backups_root)
        old = [(datetime.datetime(2020, 1, 1, tzinfo=UTC)
                + datetime.timedelta(seconds=i)).strftime(STAMP) + "-changes"
               for i in range(1005)]
        for name in old:
            os.makedirs(os.path.join(self.backups_root, name))
        self.recent_full_backup()
        self.overwrite_pair(tl.uid(5))

        self.assertEqual(self.run_cli("--execute")[0], 0)

        changes = self.changes_folders()
        self.assertEqual(len(changes), 1000)
        self.assertEqual(changes[:999], old[-999:])
        self.assertGreater(changes[-1], old[-1])

    def test_a_full_backup_with_an_empty_tree_counts(self):
        self.recent_full_backup()
        self.overwrite_pair(tl.uid(5))

        self.assertEqual(self.run_cli("--execute")[0], 0)

        self.assertEqual(len(self.full_folders()), 1)
        self.assertEqual(len(self.changes_folders()), 1)

    def test_a_full_backup_stamped_tomorrow_does_not_stop_a_full_one(self):
        tomorrow = self.full_backup(self.now() + datetime.timedelta(days=1))
        self.overwrite_pair(tl.uid(5))

        result = self.sync()

        self.assertEqual(result.applied["errors"], 0)
        self.assertEqual(len(self.full_folders()), 2)
        self.assertTrue(os.path.isdir(tomorrow))
        self.assertTrue(os.path.isdir(result.backup))
        self.assertRegex(os.path.basename(result.backup), bridge.STAMP_RE)

    def test_the_24_hour_edge_one_second_either_side(self):
        now = datetime.datetime(2026, 11, 12, 6, 0, 0, tzinfo=UTC)
        inside = self.full_backup(now - datetime.timedelta(hours=24)
                                  + datetime.timedelta(seconds=1))
        self.assertFalse(bridge.needs_full_backup(self.state_dir, now=now))
        shutil.rmtree(inside)
        self.full_backup(now - datetime.timedelta(hours=24, seconds=1))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=now))


class ClockJumpedBackTest(HostileCase):
    """After the clock went back a day every existing stamp is 'tomorrow'."""

    def test_a_full_ring_of_future_stamps_keeps_this_runs_full_backup(self):
        tomorrow = self.now() + datetime.timedelta(days=1)
        for i in range(bridge.BACKUPS_KEPT):
            self.full_backup(tomorrow + datetime.timedelta(seconds=i))
        dest = self.overwrite_pair(tl.uid(5))
        old = self.read(dest)

        result = self.sync()

        self.assertEqual(result.applied["copies_written"], 2)
        # The run overwrote dest, so its old bytes must survive somewhere.
        self.assertTrue(os.path.isdir(result.backup),
                        "the run's own backup was pruned before it applied")
        self.assertEqual(self.read(self.saved(
            os.path.basename(result.backup), dest)), old)

    def test_a_changes_ring_of_future_stamps_keeps_this_runs_changes(self):
        original = bridge.CHANGES_KEPT
        bridge.CHANGES_KEPT = 2
        self.addCleanup(setattr, bridge, "CHANGES_KEPT", original)
        self.recent_full_backup()
        tomorrow = self.now() + datetime.timedelta(days=1)
        for i in range(2):
            os.makedirs(os.path.join(
                self.backups_root,
                (tomorrow + datetime.timedelta(seconds=i)).strftime(STAMP)
                + "-changes"))
        dest = self.overwrite_pair(tl.uid(5))
        old = self.read(dest)

        result = self.sync()

        self.assertEqual(result.applied["copies_written"], 2)
        self.assertTrue(os.path.isdir(result.backup),
                        "the run's own changes folder was pruned")
        self.assertEqual(self.read(self.saved(
            os.path.basename(result.backup), dest)), old)


class TwoRingsTest(HostileCase):
    def test_both_rings_far_past_their_caps_keep_their_own_newest(self):
        fulls = ["2020%04dT000000000000Z" % (101 + i) for i in range(30)]
        changes = [(datetime.datetime(2019, 1, 1, tzinfo=UTC)
                    + datetime.timedelta(seconds=i)).strftime(STAMP)
                   + "-changes" for i in range(1100)]
        os.makedirs(self.backups_root)
        for name in fulls + changes:
            os.makedirs(os.path.join(self.backups_root, name))
        outside = os.path.join(self.base, "outside")
        os.makedirs(os.path.join(outside, "inner"))
        tl.write_other(outside, "inner/keep.json", b"outside", 1_000_000_000)
        os.chmod(os.path.join(outside, "inner"), 0o500)
        self.addCleanup(chmod_if_present, os.path.join(outside, "inner"),
                        0o700)
        links = ["19990101T000000000000Z", "29990101T000000000000Z",
                 "19990101T000000000000Z-changes",
                 "29990101T000000000000Z-changes"]
        for name in links:
            os.symlink(outside, os.path.join(self.backups_root, name))
        before_outside = tl.snapshot(outside)
        mode = stat.S_IMODE(os.lstat(os.path.join(outside, "inner")).st_mode)

        removed = bridge.prune_backups(self.backups_root)

        self.assertEqual(len(removed), 20 + 100)
        left = self.names()
        self.assertEqual([n for n in fulls if n in left], fulls[-10:])
        self.assertEqual([n for n in changes if n in left], changes[-1000:])
        for name in links:
            self.assertTrue(os.path.islink(os.path.join(self.backups_root,
                                                        name)))
        self.assertEqual(tl.snapshot(outside), before_outside)
        self.assertEqual(stat.S_IMODE(os.lstat(
            os.path.join(outside, "inner")).st_mode), mode)


# ======================================================================== logs

class LogCase(HostileCase):
    def setUp(self):
        super().setUp()
        self.logs = os.path.join(self.home, "Library", "Logs")
        os.makedirs(self.logs)
        self.out_log, self.err_log = bridge.log_paths(self.home)
        self.cap = bridge.LOG_MAX_BYTES
        self.keep = bridge.LOG_KEEP_BYTES

    def write(self, path, data):
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def lines(self, size, width=100):
        line = b"L" * (width - 1) + b"\n"
        count, rest = divmod(size, width)
        return line * count + b"r" * rest

    def assert_trimmed_tail(self, original, kept):
        self.assertLessEqual(len(kept), self.keep)
        self.assertTrue(original.endswith(kept))


class LogSizeTest(LogCase):
    def test_exactly_at_the_cap_is_untouched(self):
        data = self.lines(self.cap)
        self.write(self.out_log, data)
        mtime = os.lstat(self.out_log).st_mtime_ns

        self.assertEqual(bridge.trim_logs(), [])

        self.assertEqual(self.read(self.out_log), data)
        self.assertEqual(os.lstat(self.out_log).st_mtime_ns, mtime)

    def test_one_byte_over_the_cap_is_cut_at_a_line_start(self):
        data = self.lines(self.cap) + b"\n"
        self.write(self.out_log, data)

        self.assertEqual(bridge.trim_logs(), [self.out_log])

        kept = self.read(self.out_log)
        self.assert_trimmed_tail(data, kept)
        self.assertEqual(data[-len(kept) - 1:-len(kept)], b"\n")
        self.assertGreater(len(kept), self.keep - 100)

    def test_fifty_megabytes_is_cut_to_the_tail(self):
        data = self.lines(50 * MB, width=1000) + b"last line\n"
        self.write(self.err_log, data)

        self.assertEqual(bridge.trim_logs(), [self.err_log])

        kept = self.read(self.err_log)
        self.assert_trimmed_tail(data, kept)
        self.assertTrue(kept.endswith(b"last line\n"))
        self.assertEqual(data[-len(kept) - 1:-len(kept)], b"\n")

    def test_a_last_window_with_no_newline_keeps_the_raw_tail(self):
        data = self.lines(self.cap) + b"z" * (self.keep + 4096)
        self.write(self.out_log, data)

        bridge.trim_logs()

        self.assertEqual(self.read(self.out_log), b"z" * self.keep)

    def test_only_newlines(self):
        data = b"\n" * (self.cap + 10)
        self.write(self.out_log, data)

        bridge.trim_logs()

        self.assertEqual(self.read(self.out_log), b"\n" * self.keep)

    def test_binary_junk(self):
        data = os.urandom(self.cap + 12345)
        self.write(self.out_log, data)

        self.assertEqual(bridge.trim_logs(), [self.out_log])

        kept = self.read(self.out_log)
        self.assert_trimmed_tail(data, kept)
        window = data[-self.keep - 1:]
        cut = window.find(b"\n")
        self.assertEqual(kept, window[1:] if cut == -1 else window[cut + 1:])

    def test_a_final_line_longer_than_the_tail_ending_in_a_newline(self):
        data = self.lines(self.cap) + b"q" * (self.keep + 10) + b"\n"
        self.write(self.out_log, data)

        bridge.trim_logs()

        kept = self.read(self.out_log)
        self.assert_trimmed_tail(data, kept)

    def test_an_appending_writer_before_and_after_the_trim(self):
        self.write(self.out_log, self.lines(self.cap))
        with open(self.out_log, "ab") as writer:
            writer.write(b"before the trim\n")
            writer.flush()
            self.assertEqual(bridge.trim_logs(), [self.out_log])
            kept = self.read(self.out_log)
            writer.write(b"after the trim\n")
            writer.flush()

        final = self.read(self.out_log)
        self.assertTrue(kept.endswith(b"before the trim\n"))
        self.assertEqual(final, kept + b"after the trim\n")
        self.assertNotIn(b"\x00", final)

    def test_a_second_trim_changes_nothing(self):
        self.write(self.out_log, self.lines(self.cap + 1000))
        self.write(self.err_log, self.lines(self.cap + 1000))
        self.assertEqual(len(bridge.trim_logs()), 2)
        before = tl.snapshot(self.logs)

        self.assertEqual(bridge.trim_logs(), [])

        self.assertEqual(tl.snapshot(self.logs), before)


class LogPathShapeTest(LogCase):
    def test_a_symlink_is_never_followed(self):
        target = self.write(os.path.join(self.base, "target.log"),
                            self.lines(self.cap + 1000))
        before = (self.read(target), os.lstat(target).st_mtime_ns)
        os.symlink(target, self.out_log)

        self.assertEqual(bridge.trim_logs(), [])

        self.assertEqual((self.read(target), os.lstat(target).st_mtime_ns),
                         before)

    def test_a_directory(self):
        os.makedirs(os.path.join(self.out_log, "child"))

        self.assertEqual(bridge.trim_logs(), [])

        self.assertTrue(os.path.isdir(os.path.join(self.out_log, "child")))

    def test_a_fifo_is_never_opened(self):
        # _trim_log lstat()s before it opens; if it ever opened the FIFO for
        # read-write the call would still return (O_RDWR never blocks on a
        # FIFO), so a hang here is impossible either way.
        os.mkfifo(self.out_log)

        self.assertEqual(bridge.trim_logs(), [])

        self.assertTrue(stat.S_ISFIFO(os.lstat(self.out_log).st_mode))

    def test_a_read_only_log_is_left_alone_without_failing(self):
        data = self.lines(self.cap + 1000)
        self.write(self.out_log, data)
        os.chmod(self.out_log, 0o400)
        self.addCleanup(chmod_if_present, self.out_log, 0o600)

        self.assertEqual(bridge.trim_logs(), [])

        self.assertEqual(self.read(self.out_log), data)

    def test_missing_logs_and_missing_logs_folder(self):
        self.assertEqual(bridge.trim_logs(), [])
        shutil.rmtree(self.logs)
        self.assertEqual(bridge.trim_logs(), [])
        self.assertFalse(os.path.exists(self.logs))


class TrimLogsCommandLineTest(LogCase):
    def setUp(self):
        super().setUp()
        self.data = self.lines(self.cap + 1000)
        self.write(self.out_log, self.data)

    def test_a_held_lock_exits_2_and_leaves_logs_and_tree_alone(self):
        self.overwrite_pair(tl.uid(5))
        tree = tl.snapshot(self.root)
        logs = tl.snapshot(self.logs)

        with bridge.file_lock(self.state_dir):
            with self.assertLogs(bridge.LOGGER, level="ERROR"):
                code, _ = self.run_cli("--execute", "--trim-logs")

        self.assertEqual(code, 2)
        self.assertEqual(tl.snapshot(self.logs), logs)
        self.assertEqual(tl.snapshot(self.root), tree)

    def test_a_dry_run_never_trims(self):
        before = tl.snapshot(self.logs)

        self.assertEqual(self.run_cli("--trim-logs")[0], 0)

        self.assertEqual(tl.snapshot(self.logs), before)

    def test_a_converged_tree_still_trims(self):
        self.assertEqual(self.run_cli("--execute", "--trim-logs")[0], 0)

        self.assertLessEqual(os.lstat(self.out_log).st_size, self.keep)
        self.assertEqual(self.names(), [])

    def test_a_log_that_cannot_be_trimmed_never_fails_the_sync(self):
        os.chmod(self.out_log, 0o400)
        self.addCleanup(chmod_if_present, self.out_log, 0o600)
        self.overwrite_pair(tl.uid(5))

        code, out = self.run_cli("--execute", "--trim-logs")

        self.assertEqual(code, 0)
        self.assertIn("copies_written=2 tombstones_copied=0 records_deleted=0 skipped=0 errors=0", out)
        self.assertEqual(self.read(self.out_log), self.data)

    def test_the_agents_own_argv_parses_with_the_flag(self):
        argv = bridge.plist_dict("/x/bridge.py", self.root, [],
                                 user_data=self.user_data,
                                 state_dir=self.state_dir,
                                 home=self.home)["ProgramArguments"]

        args = bridge.build_parser().parse_args(argv[2:])

        self.assertTrue(args.execute)
        self.assertTrue(args.trim_logs)


if __name__ == "__main__":
    unittest.main()
