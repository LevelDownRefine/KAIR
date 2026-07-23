"""Unit tests for utils.utils_logger.

``log`` prints a timestamped line; ``logger_info`` wires a file + stream
logger; ``logger_print`` mirrors writes to a file and stdout.
"""
import logging
import uuid

import utils.utils_logger as logger_util


def test_log_prints_message_and_timestamp(capsys):
    logger_util.log('hello-world')
    out = capsys.readouterr().out
    assert 'hello-world' in out
    assert '-' in out  # timestamp separator


def test_logger_info_writes_to_file(tmp_path):
    name = 'test_logger_{}'.format(uuid.uuid4().hex)
    log_path = tmp_path / 'run.log'
    logger_util.logger_info(name, str(log_path))
    lg = logging.getLogger(name)
    lg.info('marker-msg')
    contents = log_path.read_text()
    assert 'marker-msg' in contents


def test_logger_print_mirrors_to_file_and_stdout(capsys, tmp_path):
    log_path = tmp_path / 'dual.log'
    lp = logger_util.logger_print(str(log_path))
    lp.write('dual-line\n')
    lp.flush()
    assert 'dual-line' in log_path.read_text()
    assert 'dual-line' in capsys.readouterr().out
