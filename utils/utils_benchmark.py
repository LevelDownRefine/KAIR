import os
import yaml

# Fixed project-relative location for the aggregated benchmark results.
BENCHMARK_YML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'benchmark.yml')


def save_benchmark(weight_name, testset_name, psnr, ssim, noise_level,
                   tol_psnr=0.05, tol_ssim=1e-3):
    """Persist aggregate PSNR/SSIM for ``weight_name``/``testset_name``/``noise_level`` into the shared benchmark YML.

    On-disk structure (three levels: weight -> test set -> noise level -> metrics)::

        dncnn_25:
          BSD68:
            25: {psnr: 29.22, ssim: 0.8278}
          set5:
            25: {psnr: 31.27, ssim: 0.8709}
        ffdnet_color_clip:
          BSD68:
            15: {psnr: ..., ssim: ...}
            25: {psnr: ..., ssim: ...}
          real_faces:
            real: {psnr: 32.20, ssim: 0.8978}

    ``noise_level`` is used verbatim as the YAML key: pass an ``int`` (e.g. ``25``)
    for synthetic Gaussian noise, or the string ``'real'`` for real-noise sets.
    A single weight may thus cover many noise levels on the same test set.
    On every write the noise-level keys are stored sorted ascending
    (ints first, string keys such as ``'real'`` last).

    Writes the entry if the weight/test-set/noise triple is absent; if it already
    exists, asserts the stored values are numerically close to the freshly
    computed ones so a regression or a wrong re-run fails loudly instead of
    silently overwriting.
    """
    data = {}
    if os.path.isfile(BENCHMARK_YML):
        with open(BENCHMARK_YML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    weight_entry = data.setdefault(weight_name, {})
    testset_entry = weight_entry.setdefault(testset_name, {})
    entry = testset_entry.get(noise_level)
    if entry is None or 'psnr' not in entry or 'ssim' not in entry:
        testset_entry[noise_level] = {'psnr': float(psnr), 'ssim': float(ssim)}
        for w in data.values():
            for ts_name, ts_entry in w.items():
                w[ts_name] = dict(sorted(
                    ts_entry.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))
        os.makedirs(os.path.dirname(BENCHMARK_YML), exist_ok=True)
        with open(BENCHMARK_YML, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print('[benchmark] wrote entry for {!r}/{!r}/{!r}: psnr={:.4f}, ssim={:.6f}'.format(
            weight_name, testset_name, noise_level, float(psnr), float(ssim)))
    else:
        stored_psnr = float(entry['psnr'])
        stored_ssim = float(entry['ssim'])
        assert abs(stored_psnr - float(psnr)) <= tol_psnr, (
            'PSNR mismatch for {!r}/{!r}/{!r}: stored {:.4f} vs new {:.4f} (tol {})'.format(
                weight_name, testset_name, noise_level, stored_psnr, float(psnr), tol_psnr))
        assert abs(stored_ssim - float(ssim)) <= tol_ssim, (
            'SSIM mismatch for {!r}/{!r}/{!r}: stored {:.6f} vs new {:.6f} (tol {})'.format(
                weight_name, testset_name, noise_level, stored_ssim, float(ssim), tol_ssim))
        print('[benchmark] verified entry for {!r}/{!r}/{!r}: psnr={:.4f}, ssim={:.6f}'.format(
            weight_name, testset_name, noise_level, stored_psnr, stored_ssim))
