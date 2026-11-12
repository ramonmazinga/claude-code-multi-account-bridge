"""Light backups: one full copy a day, only the files a run touches otherwise."""

import contextlib
import datetime
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import bridge_testlib as tl
import bridge

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 12, 6, 0, 0, tzinfo=UTC)
# The documented backup folder name; written out here so these tests pin the
# layout itself rather than whatever the module happens to use.
STAMP = "%Y%m%dT%H%M%S%fZ"


def chmod_if_present(path, mode):
    if os.path.exists(path):
        os.chmod(path, mode)


class LightCase(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="bridge-test-")
        self.addCleanup(shutil.rmtree, self.base, True)
        self.user_data = os.path.join(self.base, "userData")
        self.state_dir = os.path.join(self.base, "state")
        self.backups_root = os.path.join(self.state_dir, "backups")
        self.root = tl.make_root(self.user_data)
        self.a = tl.make_leaf(self.user_data, tl.uid(1), tl.uid(11))
        self.b = tl.make_leaf(self.user_data, tl.uid(2), tl.uid(22))

    def full_backup(self, when, with_tree=True):
        """A full backup folder stamped `when`, holding a sessions tree."""
        path = os.path.join(self.backups_root, when.strftime(STAMP))
        os.makedirs(os.path.join(path, tl.SESSIONS_DIRNAME) if with_tree
                    else path)
        return path

    def recent_full_backup(self):
        return self.full_backup(datetime.datetime.now(UTC)
                                - datetime.timedelta(hours=1))

    def names(self):
        if not os.path.isdir(self.backups_root):
            return []
        return sorted(os.listdir(self.backups_root))

    def changes_folders(self):
        return [name for name in self.names() if name.endswith("-changes")]

    def sync(self):
        return bridge.sync_once(self.root, execute=True,
                                state_dir=self.state_dir)

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bridge.main(list(args), runner=tl.RecordingRunner())
        return code, out.getvalue()

    def sync_args(self, *extra):
        return ("sync", "--user-data", self.user_data,
                "--state-dir", self.state_dir) + extra

    def read(self, path):
        with open(path, "rb") as handle:
            return handle.read()


class NeedsFullBackupTest(LightCase):
    def test_no_backups_at_all_needs_one(self):
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_23_hour_old_full_backup_is_enough(self):
        self.full_backup(NOW - datetime.timedelta(hours=23))
        self.assertFalse(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_exactly_24_hours_old_needs_a_new_one(self):
        self.full_backup(NOW - datetime.timedelta(hours=24))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_stamp_in_the_future_never_counts(self):
        self.full_backup(NOW + datetime.timedelta(minutes=1))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_changes_folder_is_not_a_full_backup(self):
        os.makedirs(os.path.join(
            self.backups_root,
            (NOW - datetime.timedelta(hours=1)).strftime(STAMP) + "-changes",
            tl.SESSIONS_DIRNAME))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_symlink_named_like_a_stamp_is_not_a_full_backup(self):
        outside = os.path.join(self.base, "outside")
        os.makedirs(os.path.join(outside, tl.SESSIONS_DIRNAME))
        os.makedirs(self.backups_root)
        os.symlink(outside, os.path.join(
            self.backups_root,
            (NOW - datetime.timedelta(hours=1)).strftime(STAMP)))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_stamp_folder_without_the_sessions_tree_is_not_a_backup(self):
        self.full_backup(NOW - datetime.timedelta(hours=1), with_tree=False)
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))

    def test_a_name_matching_the_pattern_but_not_a_date_never_counts(self):
        os.makedirs(os.path.join(self.backups_root, "20261399T000000000000Z",
                                 tl.SESSIONS_DIRNAME))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))


class BackupChangesTest(LightCase):
    def test_existing_files_are_copied_with_bytes_and_exact_mtime(self):
        record = tl.write_record(self.a, tl.uid(5), mtime_ns=5_123_456_789)
        tomb = tl.write_tombstone(self.b, tl.uid(6), mtime_ns=6_987_654_321)
        missing = os.path.join(self.b, "local_%s.json" % tl.uid(7))

        backup = bridge.backup_changes(self.root, self.state_dir,
                                       [record, tomb, missing], now=NOW)

        self.assertEqual(backup.skipped, 0)
        self.assertEqual(os.path.basename(backup.path),
                         NOW.strftime(STAMP) + "-changes")
        saved_root = os.path.join(backup.path, tl.SESSIONS_DIRNAME)
        for original in (record, tomb):
            saved = os.path.join(saved_root, os.path.relpath(original, self.root))
            self.assertEqual(self.read(saved), self.read(original))
            self.assertEqual(os.lstat(saved).st_mtime_ns,
                             os.lstat(original).st_mtime_ns)
        self.assertFalse(os.path.exists(os.path.join(
            saved_root, os.path.relpath(missing, self.root))))

    def test_nothing_existing_makes_no_directory(self):
        missing = os.path.join(self.b, "local_%s.json" % tl.uid(7))

        backup = bridge.backup_changes(self.root, self.state_dir, [missing],
                                       now=NOW)

        self.assertEqual(backup, bridge.Backup(None, 0))
        self.assertFalse(os.path.exists(self.state_dir))

    def test_a_path_outside_the_root_fails(self):
        record = tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        outside = tl.write_other(self.base, "elsewhere.json", b"{}")

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            backup = bridge.backup_changes(self.root, self.state_dir,
                                           [record, outside], now=NOW)

        self.assertIsNone(backup)
        self.assertEqual(self.changes_folders(), [])

    def test_a_symlink_is_skipped_and_never_followed(self):
        target = tl.write_other(self.base, "target.json", b"outside bytes")
        link = os.path.join(self.a, "local_%s.json" % tl.uid(5))
        os.symlink(target, link)
        record = tl.write_record(self.b, tl.uid(6), mtime_ns=6_000_000_000)

        backup = bridge.backup_changes(self.root, self.state_dir,
                                       [link, record], now=NOW)

        saved_root = os.path.join(backup.path, tl.SESSIONS_DIRNAME)
        self.assertFalse(os.path.lexists(os.path.join(
            saved_root, os.path.relpath(link, self.root))))
        self.assertTrue(os.path.isfile(os.path.join(
            saved_root, os.path.relpath(record, self.root))))

    def test_only_a_symlink_makes_no_directory(self):
        target = tl.write_other(self.base, "target.json", b"outside bytes")
        link = os.path.join(self.a, "local_%s.json" % tl.uid(5))
        os.symlink(target, link)

        backup = bridge.backup_changes(self.root, self.state_dir, [link],
                                       now=NOW)

        self.assertEqual(backup, bridge.Backup(None, 0))
        self.assertEqual(self.names(), [])

    def test_a_copy_error_removes_the_half_written_folder(self):
        first = tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        second = tl.write_record(self.b, tl.uid(6), mtime_ns=6_000_000_000)
        os.chmod(second, 0o000)
        self.addCleanup(chmod_if_present, second, 0o600)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            backup = bridge.backup_changes(self.root, self.state_dir,
                                           [first, second], now=NOW)

        self.assertIsNone(backup)
        self.assertEqual(self.changes_folders(), [])

    def test_a_failed_copy_does_not_prune_a_good_changes_folder_away(self):
        """The corpse sorts NEWEST, so pruning before removing it keeps it."""
        kept = ["20200101T0000%02d000000Z-changes" % i for i in range(2)]
        os.makedirs(self.backups_root)
        for name in kept:
            os.makedirs(os.path.join(self.backups_root, name))
        original = bridge.CHANGES_KEPT
        bridge.CHANGES_KEPT = 2
        self.addCleanup(setattr, bridge, "CHANGES_KEPT", original)
        first = tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        second = tl.write_record(self.b, tl.uid(6), mtime_ns=6_000_000_000)
        os.chmod(second, 0o000)
        self.addCleanup(chmod_if_present, second, 0o600)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            self.assertIsNone(bridge.backup_changes(
                self.root, self.state_dir, [first, second], now=NOW))

        self.assertEqual(self.changes_folders(), kept)

    def test_an_unwritable_backups_folder_fails(self):
        record = tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        os.makedirs(self.backups_root)
        os.chmod(self.backups_root, 0o500)
        self.addCleanup(chmod_if_present, self.backups_root, 0o700)

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            backup = bridge.backup_changes(self.root, self.state_dir,
                                           [record], now=NOW)

        self.assertIsNone(backup)

    def test_it_prunes_the_changes_ring(self):
        os.makedirs(self.backups_root)
        for index in range(3):
            os.makedirs(os.path.join(
                self.backups_root, "20200101T0000%02d000000Z-changes" % index))
        record = tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        original = bridge.CHANGES_KEPT
        bridge.CHANGES_KEPT = 2
        self.addCleanup(setattr, bridge, "CHANGES_KEPT", original)

        bridge.backup_changes(self.root, self.state_dir, [record], now=NOW)

        self.assertEqual(self.changes_folders(), [
            "20200101T000002000000Z-changes",
            NOW.strftime(STAMP) + "-changes"])


class PruneRingsTest(LightCase):
    def make(self, *names):
        os.makedirs(self.backups_root, exist_ok=True)
        for name in names:
            os.makedirs(os.path.join(self.backups_root, name))

    def test_each_ring_is_trimmed_on_its_own(self):
        fulls = ["20200101T0000%02d000000Z" % i for i in range(12)]
        changes = ["20200102T0000%02d000000Z-changes" % i for i in range(3)]
        self.make(*(fulls + changes))

        removed = bridge.prune_backups(self.backups_root, keep=10,
                                       changes_kept=2)

        self.assertEqual(sorted(removed), sorted(fulls[:2] + changes[:1]))
        self.assertEqual(self.names(), sorted(fulls[2:] + changes[1:]))

    def test_many_changes_never_evict_the_one_full_backup(self):
        full = "20300101T000000000000Z"
        changes = ["20200101T0000%02d000000Z-changes" % i for i in range(15)]
        self.make(full, *changes)

        bridge.prune_backups(self.backups_root, keep=10, changes_kept=5)

        self.assertIn(full, self.names())
        self.assertEqual(self.changes_folders(), changes[-5:])

    def test_many_fulls_never_evict_a_changes_folder(self):
        fulls = ["20300101T0000%02d000000Z" % i for i in range(12)]
        old_change = "20200101T000000000000Z-changes"
        self.make(old_change, *fulls)

        bridge.prune_backups(self.backups_root, keep=10, changes_kept=5)

        self.assertEqual(self.changes_folders(), [old_change])
        self.assertEqual(len([n for n in self.names()
                              if not n.endswith("-changes")]), 10)


class SyncOnceBackupChoiceTest(LightCase):
    def take_backup_calls(self):
        calls = []
        original = bridge.take_backup

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        bridge.take_backup = spy
        self.addCleanup(setattr, bridge, "take_backup", original)
        return calls

    def test_no_recent_full_backup_means_a_full_backup(self):
        tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        old = self.full_backup(NOW - datetime.timedelta(days=30))

        result = self.sync()

        self.assertEqual(self.changes_folders(), [])
        self.assertEqual(len(self.names()), 2)
        self.assertNotEqual(result.backup, old)
        self.assertTrue(os.path.isfile(os.path.join(
            result.backup, tl.SESSIONS_DIRNAME, tl.uid(1), tl.uid(11),
            "local_%s.json" % tl.uid(5))))

    def test_a_recent_full_backup_means_only_the_touched_files_are_saved(self):
        sid, doomed_id, new_id = tl.uid(5), tl.uid(6), tl.uid(7)
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        overwritten = tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                                      title="old")
        old_bytes = self.read(overwritten)
        tl.write_tombstone(self.a, doomed_id, mtime_ns=6_000_000_000)
        doomed = tl.write_record(self.b, doomed_id, mtime_ns=2_000_000_000)
        doomed_bytes = self.read(doomed)
        tl.write_record(self.a, new_id, mtime_ns=7_000_000_000)
        recent = self.recent_full_backup()

        result = self.sync()

        self.assertEqual(result.applied["errors"], 0)
        changes = self.changes_folders()
        self.assertEqual(len(changes), 1)
        self.assertEqual(result.backup,
                         os.path.join(self.backups_root, changes[0]))
        self.assertEqual(sorted(n for n in self.names()
                                if not n.endswith("-changes")),
                         [os.path.basename(recent)])
        saved = os.path.join(result.backup, tl.SESSIONS_DIRNAME)
        files = {rel: (payload, mtime) for rel, (kind, payload, mtime)
                 in tl.snapshot(saved).items() if kind != "dir"}
        leaf = tl.uid(2) + "/" + tl.uid(22) + "/"
        self.assertEqual(files, {
            leaf + "local_%s.json" % sid: (old_bytes, 1_000_000_000),
            leaf + "local_%s.json" % doomed_id: (doomed_bytes, 2_000_000_000),
        })
        # ... and the run itself still applied everything
        self.assertFalse(os.path.exists(doomed))
        self.assertTrue(os.path.exists(
            os.path.join(self.b, "local_%s.json" % new_id)))
        self.assertTrue(os.path.exists(
            os.path.join(self.b, "deleted_%s" % doomed_id)))

    def test_pure_additions_need_no_backup_folder_at_all(self):
        tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        recent = self.recent_full_backup()

        code, out = self.run_cli(*self.sync_args("--execute"))

        self.assertEqual(code, 0)
        self.assertEqual(self.names(), [os.path.basename(recent)])
        self.assertTrue(os.path.exists(
            os.path.join(self.b, "local_%s.json" % tl.uid(5))))
        self.assertEqual([line for line in out.splitlines()
                          if line.startswith("backup:")], [])

    def test_pure_additions_report_no_backup_from_sync_once(self):
        tl.write_record(self.a, tl.uid(5), mtime_ns=5_000_000_000)
        self.recent_full_backup()

        result = self.sync()

        self.assertIsNone(result.backup)
        self.assertEqual(result.applied["copies_written"], 1)

    def test_a_failed_changes_backup_applies_nothing(self):
        sid = tl.uid(5)
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        overwritten = tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                                      title="old")
        before = tl.snapshot(self.root)
        self.recent_full_backup()
        os.chmod(self.backups_root, 0o500)
        self.addCleanup(chmod_if_present, self.backups_root, 0o700)
        full_calls = self.take_backup_calls()

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            code, out = self.run_cli(*self.sync_args("--execute"))

        self.assertEqual(code, 1)
        self.assertEqual(full_calls, [], "a full backup < 24 h old exists")
        self.assertIn("errors=1", out)
        self.assertEqual(tl.snapshot(self.root), before)
        self.assertEqual(self.changes_folders(), [])
        self.assertTrue(os.path.exists(overwritten))

    def test_a_failed_changes_backup_sets_one_error(self):
        sid = tl.uid(5)
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        tl.write_record(self.b, sid, mtime_ns=1_000_000_000, title="old")
        self.recent_full_backup()
        os.chmod(self.backups_root, 0o500)
        self.addCleanup(chmod_if_present, self.backups_root, 0o700)
        full_calls = self.take_backup_calls()

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            result = self.sync()

        self.assertEqual(full_calls, [])
        self.assertEqual(result.applied, bridge.empty_applied(errors=1))
        self.assertIsNone(result.backup)

    def test_the_backup_line_names_the_changes_folder(self):
        sid = tl.uid(5)
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        tl.write_record(self.b, sid, mtime_ns=1_000_000_000, title="old")
        self.recent_full_backup()

        code, out = self.run_cli(*self.sync_args("--execute"))

        self.assertEqual(code, 0)
        changes = self.changes_folders()
        self.assertEqual(len(changes), 1)
        self.assertEqual([line for line in out.splitlines()
                          if line.startswith("backup:")],
                         ["backup: %s skipped=0" % os.path.join(
                             self.backups_root, changes[0])])


PARTIAL = ".partial-"


class ThisRunsBackupSurvivesTest(LightCase):
    """F1/F2: a ring of future stamps must not prune the backup just made."""

    def overwrite_pair(self, sid):
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        return tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                               title="old")

    def test_1000_future_changes_folders_keep_this_runs_change_set(self):
        self.recent_full_backup()
        tomorrow = datetime.datetime.now(UTC) + datetime.timedelta(days=1)
        for i in range(1000):
            os.makedirs(os.path.join(
                self.backups_root,
                (tomorrow + datetime.timedelta(seconds=i)).strftime(STAMP)
                + "-changes"))
        dest = self.overwrite_pair(tl.uid(5))
        old = self.read(dest)

        result = self.sync()

        self.assertEqual(result.applied["copies_written"], 1)
        self.assertNotEqual(self.read(dest), old)
        self.assertTrue(os.path.isdir(result.backup))
        self.assertEqual(self.read(os.path.join(
            result.backup, tl.SESSIONS_DIRNAME,
            os.path.relpath(dest, self.root))), old)

    def test_10_future_full_backups_keep_this_runs_full_copy(self):
        now = datetime.datetime.now(UTC)
        for i in range(1, 11):
            self.full_backup(now + datetime.timedelta(days=i))
        dest = self.overwrite_pair(tl.uid(5))
        old = self.read(dest)

        result = self.sync()

        self.assertEqual(result.applied["copies_written"], 1)
        self.assertTrue(os.path.isdir(result.backup))
        self.assertEqual(self.read(os.path.join(
            result.backup, tl.SESSIONS_DIRNAME,
            os.path.relpath(dest, self.root))), old)


class PruneProtectTest(LightCase):
    def make(self, *names):
        os.makedirs(self.backups_root, exist_ok=True)
        for name in names:
            os.makedirs(os.path.join(self.backups_root, name))

    def test_a_protected_oldest_full_backup_is_kept(self):
        fulls = ["20200101T0000%02d000000Z" % i for i in range(12)]
        changes = ["20200102T0000%02d000000Z-changes" % i for i in range(3)]
        self.make(*(fulls + changes))

        removed = bridge.prune_backups(self.backups_root, keep=10,
                                       changes_kept=5, protect=(fulls[0],))

        self.assertEqual(sorted(removed), fulls[1:3])
        self.assertEqual(self.names(), sorted([fulls[0]] + fulls[3:] + changes))

    def test_a_protected_oldest_changes_folder_is_kept(self):
        fulls = ["20200101T0000%02d000000Z" % i for i in range(3)]
        changes = ["20200102T0000%02d000000Z-changes" % i for i in range(7)]
        self.make(*(fulls + changes))

        removed = bridge.prune_backups(self.backups_root, keep=10,
                                       changes_kept=5, protect=(changes[0],))

        self.assertEqual(sorted(removed), changes[1:3])
        self.assertEqual(self.names(),
                         sorted(fulls + [changes[0]] + changes[3:]))


class InterruptedBackupTest(LightCase):
    """F3: a copy killed partway leaves nothing under a final backup name."""

    def interrupt_after(self, count):
        real = shutil.copyfile
        calls = []

        def copyfile(*args, **kwargs):
            calls.append(args[0])
            if len(calls) > count:
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        patcher = mock.patch.object(shutil, "copyfile", copyfile)
        patcher.start()
        return patcher

    def overwrite_pairs(self, count):
        dests = []
        for i in range(count):
            sid = tl.uid(50 + i)
            tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
            dests.append(tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                                         title="old"))
        return dests

    def test_an_interrupted_full_copy_is_not_trusted(self):
        self.overwrite_pairs(4)
        patcher = self.interrupt_after(3)
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.sync()
        finally:
            patcher.stop()

        self.assertEqual([n for n in self.names()
                          if bridge.STAMP_RE.match(n)], [])
        self.assertTrue(bridge.needs_full_backup(self.state_dir))

        result = self.sync()

        self.assertEqual(result.applied["errors"], 0)
        self.assertEqual(self.names(), [os.path.basename(result.backup)])
        saved = tl.snapshot(os.path.join(result.backup, tl.SESSIONS_DIRNAME))
        self.assertEqual(len([k for k, (kind, _, _) in saved.items()
                              if kind != "dir"]), 8)

    def test_an_interrupted_change_set_is_not_left_under_its_name(self):
        self.overwrite_pairs(4)
        recent = self.recent_full_backup()
        patcher = self.interrupt_after(3)
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.sync()
        finally:
            patcher.stop()

        self.assertEqual(self.changes_folders(), [])

        result = self.sync()

        self.assertEqual(result.applied["errors"], 0)
        self.assertEqual(self.names(), sorted(
            [os.path.basename(recent), os.path.basename(result.backup)]))


class PartialLeftoversTest(LightCase):
    def test_prune_removes_real_partial_leftovers_but_not_a_symlink(self):
        os.makedirs(self.backups_root)
        full = PARTIAL + "20200101T000000000000Z"
        changes = PARTIAL + "20200101T000001000000Z-changes"
        for name in (full, changes):
            os.makedirs(os.path.join(self.backups_root, name,
                                     tl.SESSIONS_DIRNAME))
        outside = os.path.join(self.base, "outside")
        tl.write_other(outside, "keep.json", b"outside")
        link = PARTIAL + "20200101T000002000000Z"
        os.symlink(outside, os.path.join(self.backups_root, link))
        before = tl.snapshot(outside)

        removed = bridge.prune_backups(self.backups_root)

        self.assertEqual(sorted(removed), sorted([full, changes]))
        self.assertEqual(self.names(), [link])
        self.assertEqual(tl.snapshot(outside), before)

    def assert_left_alone(self, *names):
        for name in names:
            os.makedirs(os.path.join(self.backups_root, name))
        self.assertEqual(bridge.prune_backups(self.backups_root), [])
        self.assertEqual(self.names(), sorted(names))

    def test_a_user_folder_named_partial_something_is_left_alone(self):
        self.assert_left_alone(PARTIAL + "my-notes")

    def test_a_folder_named_just_the_partial_prefix_is_left_alone(self):
        self.assert_left_alone(PARTIAL)

    def test_a_partial_folder_with_an_impossible_stamp_is_left_alone(self):
        self.assert_left_alone(PARTIAL + "20261345T250000000000Z",
                               PARTIAL + "20261345T250000000000Z-changes")

    def test_a_user_partial_folder_takes_no_slot(self):
        full = "20200101T000000000000Z"
        self.assert_left_alone(full, PARTIAL + "my-notes")
        self.assertEqual(bridge.prune_backups(self.backups_root, keep=1), [])
        self.assertEqual(self.names(), sorted([full, PARTIAL + "my-notes"]))

    def test_a_partial_full_copy_does_not_count_as_a_full_backup(self):
        os.makedirs(os.path.join(
            self.backups_root,
            PARTIAL + (NOW - datetime.timedelta(minutes=1)).strftime(STAMP),
            tl.SESSIONS_DIRNAME))
        self.assertTrue(bridge.needs_full_backup(self.state_dir, now=NOW))


class FailedRenameTest(LightCase):
    def fail_partial_renames(self):
        real = os.rename

        def rename(src, dst, *args, **kwargs):
            if os.path.basename(src).startswith(PARTIAL):
                raise OSError("synthetic rename failure")
            return real(src, dst, *args, **kwargs)

        patcher = mock.patch.object(os, "rename", rename)
        patcher.start()
        self.addCleanup(patcher.stop)

    def overwrite(self):
        sid = tl.uid(5)
        tl.write_record(self.a, sid, mtime_ns=9_000_000_000, title="new")
        return tl.write_record(self.b, sid, mtime_ns=1_000_000_000,
                               title="old")

    def assert_nothing_applied(self, extra_names):
        before = tl.snapshot(self.root)
        self.fail_partial_renames()

        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            result = self.sync()

        self.assertIsNone(result.backup)
        self.assertEqual(result.applied["errors"], 1)
        self.assertEqual(tl.snapshot(self.root), before)
        self.assertEqual(self.names(), extra_names)

    def test_a_failed_rename_of_a_full_copy_applies_nothing(self):
        self.overwrite()
        self.assert_nothing_applied([])

    def test_a_failed_rename_of_a_change_set_applies_nothing(self):
        self.overwrite()
        recent = self.recent_full_backup()
        self.assert_nothing_applied([os.path.basename(recent)])

    def test_take_backup_returns_none_on_a_failed_rename(self):
        self.overwrite()
        self.fail_partial_renames()
        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            self.assertIsNone(bridge.take_backup(self.root, self.state_dir,
                                                 now=NOW))
        self.assertEqual(self.names(), [])

    def test_backup_changes_returns_none_on_a_failed_rename(self):
        dest = self.overwrite()
        self.fail_partial_renames()
        with self.assertLogs(bridge.LOGGER, level="ERROR"):
            self.assertIsNone(bridge.backup_changes(
                self.root, self.state_dir, [dest], now=NOW))
        self.assertEqual(self.names(), [])


if __name__ == "__main__":
    unittest.main()
