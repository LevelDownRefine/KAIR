"""Unit tests for utils.utils_option.

Covers the configuration helpers with numerical / structural assertions:
timestamp formatting, checkpoint discovery, the opt parser's default
broadcasts, dict->string rendering and the NoneDict fallback.
"""
import os
import re
import json
import tempfile

import utils.utils_option as option


def test_get_timestamp_format():
    ts = option.get_timestamp()
    assert re.fullmatch(r'_\d{6}_\d{6}', ts), ts


def test_find_last_checkpoint_empty_dir_returns_pretrained():
    with tempfile.TemporaryDirectory() as d:
        init_iter, init_path = option.find_last_checkpoint(d, net_type='G', pretrained_path='fallback.pth')
        assert init_iter == 0
        assert init_path == 'fallback.pth'


def test_find_last_checkpoint_picks_max_iter():
    with tempfile.TemporaryDirectory() as d:
        for i in (100, 50, 250, 12):
            open(os.path.join(d, '{}_G.pth'.format(i)), 'w').close()
        init_iter, init_path = option.find_last_checkpoint(d, net_type='G')
        assert init_iter == 250
        assert init_path == os.path.join(d, '250_G.pth')


def _write_minimal_opt(path, is_train=True):
    opt = {
        'task': 'sr',
        'scale': 2,
        'n_channels': 3,
        'gpu_ids': [0],
        'datasets': {'train': {'dataroot_H': None, 'dataroot_L': None},
                     'test': {'dataroot_H': None}},
        'path': {'root': '/tmp/exp', 'options': '/tmp/exp/options'},
        'netG': {},
        'train': {},
    }
    with open(path, 'w') as f:
        json.dump(opt, f)
    return path


def test_parse_applies_defaults_and_broadcasts():
    with tempfile.TemporaryDirectory() as d:
        opt_path = os.path.join(d, 'opt.json')
        _write_minimal_opt(opt_path)
        opt = option.parse(opt_path, is_train=True)

        # defaults
        assert opt['merge_bn'] is False
        assert opt['scale'] == 2
        # dataset broadcasts
        assert opt['datasets']['train']['scale'] == 2
        assert opt['datasets']['train']['n_channels'] == 3
        assert opt['datasets']['train']['phase'] == 'train'
        assert opt['datasets']['test']['phase'] == 'test'
        # GPU / distributed defaults
        assert opt['num_gpu'] == 1
        assert opt['dist'] is False
        # path.task derived from root + task
        assert opt['path']['task'].endswith('sr')
        assert opt['path']['models'].endswith('models')
        # CUDA env side-effect
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'


def test_dict2str_renders_nested_dict():
    d = {'a': 1, 'b': {'c': 2}}
    s = option.dict2str(d)
    assert 'a: 1' in s
    assert 'b:[' in s
    assert 'c: 2' in s


def test_dict_to_nonedict_missing_key_is_none():
    nd = option.dict_to_nonedict({'x': {'y': 1}})
    assert nd['x']['y'] == 1
    assert nd['x']['missing'] is None
    assert nd['missing'] is None
