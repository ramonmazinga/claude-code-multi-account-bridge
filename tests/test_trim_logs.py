"""The agent's two log files are trimmed in place, never past the cap."""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import bridge_testlib as tl
import bridge


class TrimCase(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="bridge-test-")
        self.addCleanup(shutil.rmtree, self.base, True)
        # NEVER the real ~/Library/Logs.
        self.home = os.path.join(self.base, "home")
        self.out_log, self.err_log = bridge.log_paths(self.home)
        os.makedirs(os.path.dirname(self.out_log))
        for name, value in (("LOG_MAX_BYTES", 1000), ("LOG_KEEP_BYTES", 100)):
            original = getattr(bridge, name)
            setattr(bridge, name, value)
            self.addCleanup(setattr, bridge, name, original)

    def write(self, path, data):
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def read(self, path):
        with open(path, "rb") as handle:
            return handle.read()

    def lines(self, count):
        return b"".join(b"%019d\n" % i for i in range(count))


class TrimLogsTest(TrimCase):
    def test_a_log_over_the_cap_keeps_its_newest_whole_lines(self):
        data = self.lines(60)  # 1200 bytes, 20 per line
        self.write(self.out_log, data)

        trimmed = bridge.trim_logs(self.home)

        self.assertEqual(trimmed, [self.out_log])
        kept = self.read(self.out_log)
        self.assertLessEqual(len(kept), 100)
        self.assertTrue(data.endswith(kept))
        # starts at a line start: the byte before it in the original is "\n"
        self.assertEqual(data[len(data) - len(kept) - 1:len(data) - len(kept)],
                         b"\n")
        self.assertEqual(kept, self.lines(60)[-100:])

    def test_a_tail_that_starts_mid_line_drops_the_partial_line(self):
        data = b"x" * 950 + b"".join(b"%014d\n" % i for i in range(10))
        self.write(self.err_log, data)

        bridge.trim_logs(self.home)

        kept = self.read(self.err_log)
        self.assertLess(len(kept), 100)
        self.assertTrue(data.endswith(kept))
        self.assertEqual(data[len(data) - len(kept) - 1:len(data) - len(kept)],
                         b"\n")

    def test_a_log_under_the_cap_is_untouched(self):
        data = self.lines(40)  # 800 bytes
        self.write(self.out_log, data)
        mtime = os.lstat(self.out_log).st_mtime_ns

        self.assertEqual(bridge.trim_logs(self.home), [])

        self.assertEqual(self.read(self.out_log), data)
        self.assertEqual(os.lstat(self.out_log).st_mtime_ns, mtime)

    def test_missing_logs_are_ignored(self):
        self.assertEqual(bridge.trim_logs(self.home), [])
        self.assertFalse(os.path.exists(self.out_log))
        self.assertFalse(os.path.exists(self.err_log))

    def test_a_symlink_over_the_cap_is_never_touched(self):
        target = self.write(os.path.join(self.base, "target.log"),
                            self.lines(60))
        os.symlink(target, self.out_log)

        self.assertEqual(bridge.trim_logs(self.home), [])

        self.assertTrue(os.path.islink(self.out_log))
        self.assertEqual(self.read(target), self.lines(60))

    def test_a_tail_with_no_newline_is_kept_whole(self):
        data = b"y" * 1200
        self.write(self.out_log, data)

        bridge.trim_logs(self.home)

        self.assertEqual(self.read(self.out_log), b"y" * 100)

    def test_an_appending_writer_continues_right_after_the_kept_tail(self):
        """launchd holds the log open in append mode while the agent runs."""
        self.write(self.out_log, self.lines(60))
        with open(self.out_log, "ab") as writer:
            bridge.trim_logs(self.home)
            kept = self.read(self.out_log)
            writer.write(b"after\n")
            writer.flush()

        self.assertEqual(self.read(self.out_log), kept + b"after\n")
        self.assertEqual(os.lstat(self.out_log).st_size, len(kept) + 6)

    def test_both_logs_are_trimmed(self):
        self.write(self.out_log, self.lines(60))
        self.write(self.err_log, self.lines(60))

        self.assertEqual(sorted(bridge.trim_logs(self.home)),
                         sorted([self.out_log, self.err_log]))


class TrimLogsFlagTest(TrimCase):
    def setUp(self):
        super().setUp()
        self.user_data = os.path.join(self.base, "userData")
        self.state_dir = os.path.join(self.base, "state")
        tl.make_root(self.user_data)
        self.write(self.out_log, self.lines(60))
        patcher = mock.patch.dict(os.environ, {"HOME": self.home})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bridge.main(["sync", "--user-data", self.user_data,
                                "--state-dir", self.state_dir] + list(extra),
                               runner=tl.RecordingRunner())
        return code

    def test_execute_with_the_flag_trims(self):
        self.assertEqual(self.run_cli("--execute", "--trim-logs"), 0)
        self.assertLessEqual(os.lstat(self.out_log).st_size, 100)

    def test_the_flag_without_execute_does_nothing(self):
        self.assertEqual(self.run_cli("--trim-logs"), 0)
        self.assertEqual(self.read(self.out_log), self.lines(60))

    def test_execute_without_the_flag_does_nothing(self):
        self.assertEqual(self.run_cli("--execute"), 0)
        self.assertEqual(self.read(self.out_log), self.lines(60))

    def test_trimming_happens_while_the_lock_is_held(self):
        seen = []
        original = bridge.trim_logs

        def spy(home=None):
            seen.append(os.path.exists(
                os.path.join(self.state_dir, bridge.LOCK_FILENAME)))
            return original(home)

        bridge.trim_logs = spy
        self.addCleanup(setattr, bridge, "trim_logs", original)

        self.assertEqual(self.run_cli("--execute", "--trim-logs"), 0)
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
