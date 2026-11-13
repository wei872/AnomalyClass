"""一条命令跑完整个流程 (检查数据 -> 训练检测器 -> kNN -> 融合 -> 校准 -> 训练分类器 -> 评估), 以及一条命令推理.

最常用的两条:
    python run.py                                # 训练: 默认 full 预设, 全流程, 已完成的步骤自动跳过 (中断后再跑一遍即可续上)
    python run.py infer --input <图片/目录/视频>   # 推理: 自动用上面训练出来的最终模型和阈值

其他:
    python run.py --preset smoke                 # 几十秒~几分钟的冒烟: 真实 backbone, 极少 iter/图片, 先验证环境/路径/权重 OK 再开长训练
    python run.py --preset fast                  # 免训练检测器 (kNN) + 分类器, 全程几分钟~十几分钟
    python run.py --dry-run                      # 只打印每一步会执行的命令, 不运行
    python run.py --from calibrate               # 从某一步开始重跑 (含之后各步); --steps knn,calibrate 只跑指定步骤; --force 全部重跑
    python run.py --iters 5000 --device cuda     # 常用覆盖; 其余任何子脚本参数用 --set 步骤.参数=值, 可多次:
    python run.py --set detector.total-iters=20000 --set detector.enc-fp16=true --set classifier.backbone=dinov2reg_vit_base_14

步骤 (--steps / --from 用的名字): check, detector, knn, fuse, calibrate, classifier, evaluate
每一步就是调用 scripts/ 下对应脚本 (和以前手敲的命令完全一样), 想单独调试哪一步, 把 --dry-run 打印出的命令复制出来跑即可.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = ['check', 'detector', 'knn', 'fuse', 'calibrate', 'classifier', 'evaluate']
OPTIONAL = {'check', 'fuse'}          # 失败不终止整个流程 (退化为不用它)

# ---- 预设: 每一步传给对应脚本的参数 (键 = 命令行参数名去掉 --, 值 True = 开关) ----
# full 里 detector 的参数就是 README「经线/纬线细长缺陷多、布匹方向固定」的推荐配方 (10000 iter + 早停).
PRESETS = {
    'full': {
        'steps': STEPS,
        'check': {},
        'detector': {'total-iters': 10000, 'eval-interval': 2500, 'save-interval': 1000, 'early-stop': 2,
                     'no-rot90': True, 'local-loss-w': 1.0, 'hard-pos-ratio': 0.2, 'hard-normal-w': 0.5,
                     'local-loss-warmup': 2000, 'defect-bg-w': 0.5},
        'knn': {'layer-sets': '5,7,9;3,5,7', 'bank-size': 20000, 'max-images': 300},
        'fuse': {},
        'calibrate': {'sweep': True},
        'classifier': {},
        'evaluate': {'max-images': 300},
    },
    'fast': {
        'steps': ['check', 'knn', 'calibrate', 'classifier', 'evaluate'],
        'check': {},
        'knn': {'layer-sets': '5,7,9;3,5,7', 'bank-size': 20000, 'max-images': 300},
        'calibrate': {'sweep': True},
        'classifier': {},
        'evaluate': {'max-images': 300},
    },
    'smoke': {
        'steps': STEPS,
        'check': {},
        'detector': {'total-iters': 40, 'batch-size': 4, 'eval-interval': 20, 'save-interval': 20,
                     'eval-max-images': 8, 'no-rot90': True},
        'knn': {'layer-sets': '5,7,9', 'bank-size': 2000, 'max-images': 12, 'eval-max-images': 8},
        'fuse': {'max-images': 8},
        'calibrate': {'max-images': 8, 'sweep': True},
        'classifier': {'head-epochs': 20, 'views': 2, 'min-per-class': 20},
        'evaluate': {'max-images': 8},
    },
}

# 哪些脚本接受 --root / --device
TAKES_ROOT = {'check', 'detector', 'knn', 'fuse', 'calibrate', 'classifier', 'evaluate'}
TAKES_DEVICE = {'detector', 'knn', 'fuse', 'calibrate', 'classifier', 'evaluate'}
SCRIPT = {'check': 'check_images.py', 'detector': 'train_detector.py', 'knn': 'fit_knn.py', 'fuse': 'fuse_detectors.py',
          'calibrate': 'calibrate.py', 'classifier': 'train_classifier.py', 'evaluate': 'evaluate.py'}


def flags_to_args(d):
    out = []
    for k, v in d.items():
        if v is True:
            out.append(f'--{k}')
        elif v is False or v is None:
            continue
        else:
            out += [f'--{k}', str(v)]
    return out


class Paths:
    def __init__(self, out):
        self.out = out
        self.det = os.path.join(out, 'detector', 'model_best.pth')
        self.knn = os.path.join(out, 'knn', 'model_best.pth')
        self.fused = os.path.join(out, 'fused', 'model_best.pth')
        self.cls = os.path.join(out, 'classifier', 'best.pt')
        self.final = os.path.join(out, 'final.json')

    def marker(self, step):
        return os.path.join(self.out, f'.done_{step}')


def choose_final(P):
    """融合 w 严格在 (0,1) 才用融合; w=1 用 Dinomaly, w=0 用 kNN; 没有融合就用有的那个."""
    if os.path.exists(P.fused):
        try:
            sys.path.insert(0, ROOT)
            from src.util import torch_load
            w = float(torch_load(P.fused, map_location='cpu')['hparams']['fusion_w'])
            if 0 < w < 1:
                return P.fused, f'融合检测器 (w_dinomaly={w:.2f})'
            if w >= 1 and os.path.exists(P.det):
                return P.det, '纯 Dinomaly (融合无增益)'
            if w <= 0 and os.path.exists(P.knn):
                return P.knn, '纯 kNN (kNN 单独最好)'
        except Exception as e:
            print(f'[run] 读取融合结果失败 ({e}), 退回单个检测器')
    if os.path.exists(P.det):
        return P.det, 'Dinomaly'
    if os.path.exists(P.knn):
        return P.knn, 'kNN'
    return None, None


def build_cmd(step, P, args, flags):
    common = ['--data-yaml', args.data_yaml]
    if args.root and step in TAKES_ROOT:
        common += ['--root', args.root]
    if args.device and step in TAKES_DEVICE:
        common += ['--device', args.device]
    base = [PY, os.path.join(ROOT, 'scripts', SCRIPT[step])] + common
    f = dict(flags)
    if step == 'check':                       # 坏图扫描 + train/val 近重复(数据泄漏)检查, 都只报告不改数据
        leak = [PY, os.path.join(ROOT, 'scripts', 'check_leakage.py'), '--data-yaml', args.data_yaml,
                '--max-images', '300'] + (['--root', args.root] if args.root else [])
        return [base + flags_to_args(f), leak]
    if step == 'detector':
        base += ['--out', os.path.join(P.out, 'detector')]
    elif step == 'knn':
        base += ['--out', os.path.join(P.out, 'knn')]
    elif step == 'fuse':
        base += ['--dino-ckpt', P.det, '--knn-ckpt', P.knn, '--out', P.fused]
    elif step == 'classifier':
        base += ['--out', os.path.join(P.out, 'classifier')]
    elif step == 'calibrate':
        det, _ = choose_final(P)
        det = det or P.det
        base += ['--det-ckpt', det, '--out', os.path.join(os.path.dirname(det), 'thresholds.json')]
    elif step == 'evaluate':
        det, _ = choose_final(P)
        det = det or P.det
        return [base + ['--mode', 'gt', '--cls-ckpt', P.cls] + flags_to_args({k: v for k, v in f.items() if k != 'max-images'}),
                base + ['--mode', 'pipeline', '--top-k', '0', '--det-ckpt', det, '--cls-ckpt', P.cls,
                        '--thresholds', os.path.join(os.path.dirname(det), 'thresholds.json')] + flags_to_args(f)]
    return [base + flags_to_args(f)]


def run_cmds(cmds, dry):
    for c in cmds:
        print('  $ ' + ' '.join(shlex.quote(x) for x in c), flush=True)
        if dry:
            continue
        r = subprocess.run(c, cwd=ROOT)
        if r.returncode != 0:
            return r.returncode
    return 0


def parse_sets(sets):
    out = {}
    for s in sets or []:
        if '=' not in s or '.' not in s.split('=', 1)[0]:
            raise SystemExit(f'--set 格式应为 步骤.参数=值, 如 detector.total-iters=5000, 收到: {s}')
        key, val = s.split('=', 1)
        step, flag = key.split('.', 1)
        if step not in STEPS:
            raise SystemExit(f'--set 里的步骤名 {step!r} 不对, 可选: {", ".join(STEPS)}')
        flag = flag.replace('_', '-')
        v = {'true': True, 'false': False}.get(val.lower(), val)
        out.setdefault(step, {})[flag] = v
    return out


def cmd_train(args):
    preset = PRESETS[args.preset]
    P = Paths(args.out)
    # 数据路径提前检查, 避免跑到一半才发现路径不对
    sys.path.insert(0, ROOT)
    from src.data import load_fabric_config
    cfg = load_fabric_config(args.data_yaml, root_override=args.root)
    print(cfg.summary())
    missing = [(k, v) for k, v in [('train', cfg.train_images), ('val', cfg.val_images),
                                   ('normaltrain', cfg.normaltrain_images), ('normalval', cfg.normalval_images)]
               if not v or not os.path.isdir(v)]
    if missing:
        raise SystemExit('数据目录不存在, 请先改 dataset/fabric19.yaml (或用 --root):\n' +
                         '\n'.join(f'  {k}: {v}' for k, v in missing))

    steps = [s for s in preset['steps']]
    if args.steps:
        steps = [s.strip() for s in args.steps.split(',') if s.strip()]
        bad = [s for s in steps if s not in STEPS]
        if bad:
            raise SystemExit(f'未知步骤 {bad}, 可选: {", ".join(STEPS)}')
    elif args.from_step:
        if args.from_step not in STEPS:
            raise SystemExit(f'--from 未知步骤 {args.from_step}, 可选: {", ".join(STEPS)}')
        steps = [s for s in STEPS if STEPS.index(s) >= STEPS.index(args.from_step) and s in preset['steps']]
    user = parse_sets(args.set)
    if args.iters:
        user.setdefault('detector', {})['total-iters'] = args.iters
    if args.no_classifier and 'classifier' in steps:
        steps.remove('classifier')

    print(f'\n[run] 预设={args.preset}  输出目录={args.out}  步骤: {" -> ".join(steps)}'
          f'{"  (dry-run)" if args.dry_run else ""}')
    os.makedirs(args.out, exist_ok=True)
    times = {}
    for step in steps:
        if args.force or args.steps or args.from_step:
            skip = False
        else:
            skip = os.path.exists(P.marker(step))
        if step == 'fuse' and not (os.path.exists(P.det) and os.path.exists(P.knn)) and not args.dry_run:
            print(f'\n=== [{step}] 跳过: 需要 detector 与 knn 两个检测器都已训练 ===')
            continue
        if skip:
            print(f'\n=== [{step}] 已完成, 跳过 (要重跑加 --force 或 --from {step}) ===')
            continue
        flags = dict(preset.get(step, {}))
        flags.update(user.get(step, {}))
        cmds = build_cmd(step, P, args, flags)
        print(f'\n=== [{step}] ===', flush=True)
        t0 = time.time()
        rc = run_cmds(cmds, args.dry_run)
        times[step] = time.time() - t0
        if rc != 0:
            if step in OPTIONAL:
                print(f'[run] 步骤 {step} 失败 (退出码 {rc}), 它是可选步骤, 继续后面的流程')
                continue
            print(f'\n[run] 步骤 {step} 失败 (退出码 {rc}). 修复后重新运行 `python run.py` 即可: 已完成的步骤会自动跳过.')
            raise SystemExit(rc)
        if not args.dry_run:
            open(P.marker(step), 'w').write(time.strftime('%F %T'))
    if args.dry_run:
        return
    det, desc = choose_final(P)
    final = {'det_ckpt': det, 'thresholds': os.path.join(os.path.dirname(det), 'thresholds.json') if det else None,
             'cls_ckpt': P.cls if os.path.exists(P.cls) else None, 'detector': desc, 'preset': args.preset}
    json.dump(final, open(P.final, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n' + '=' * 60)
    for s, t in times.items():
        print(f'  {s:<11s} {t / 60:6.1f} min')
    print(f'最终检测器: {desc}\n  {det}\n分类器: {final["cls_ckpt"]}\n记录: {P.final}')
    print('\n推理 (一条命令):\n  python run.py infer --input <图片/目录/视频>')
    print('=' * 60)


def cmd_infer(args, rest):
    P = Paths(args.out)
    if not os.path.exists(P.final):
        raise SystemExit(f'找不到 {P.final}: 请先 `python run.py` 训练 (或用 --out 指向训练时的输出目录)')
    fin = json.load(open(P.final, encoding='utf-8'))
    cmd = [PY, os.path.join(ROOT, 'scripts', 'infer.py'), '--data-yaml', args.data_yaml,
           '--det-ckpt', fin['det_ckpt'], '--thresholds', fin['thresholds'], '--out', args.save]
    if args.root:
        cmd += ['--root', args.root]
    if args.device:
        cmd += ['--device', args.device]
    if fin.get('cls_ckpt') and not args.no_classify:
        cmd += ['--cls-ckpt', fin['cls_ckpt']]
    else:
        cmd += ['--no-classify']
    inp = args.input
    if inp is None:
        raise SystemExit('请用 --input 指定图片 / 目录 / 视频')
    if os.path.isdir(inp):
        cmd += ['--dir', inp]
    elif inp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cmd += ['--video', inp]
    elif inp.isdigit():
        cmd += ['--camera', inp]
    else:
        cmd += ['--image', inp]
    cmd += rest
    print(f'[run] 检测器: {fin["detector"]}')
    print('  $ ' + ' '.join(shlex.quote(x) for x in cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)


def main():
    argv = sys.argv[1:]
    mode = 'train'
    if argv and argv[0] in ('train', 'infer'):
        mode, argv = argv[0], argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-yaml', default=os.path.join(ROOT, 'dataset', 'fabric19.yaml'))
    p.add_argument('--root', default=None, help='数据集根目录覆盖 (也可用环境变量 FABRIC19_ROOT)')
    p.add_argument('--out', default=os.path.join(ROOT, 'checkpoints'), help='模型/阈值输出根目录')
    p.add_argument('--device', default=None, help='cuda / cpu (默认各脚本自动选)')
    # train
    p.add_argument('--preset', default='full', choices=sorted(PRESETS), help='full=全流程(默认) / fast=免训练检测器 / smoke=冒烟')
    p.add_argument('--steps', default=None, help='只跑这些步骤 (逗号分隔), 如 knn,calibrate')
    p.add_argument('--from', dest='from_step', default=None, help='从这一步开始重跑 (含之后各步)')
    p.add_argument('--force', action='store_true', help='忽略已完成标记, 全部重跑')
    p.add_argument('--iters', type=int, default=None, help='检测器训练 iter 数 (覆盖预设)')
    p.add_argument('--no-classifier', action='store_true', help='不训练分类器')
    p.add_argument('--set', action='append', default=[], metavar='步骤.参数=值', help='覆盖/追加子脚本参数, 可多次')
    p.add_argument('--dry-run', action='store_true', help='只打印命令')
    # infer
    p.add_argument('--input', default=None, help='[infer] 图片 / 目录 / 视频 / 摄像头编号')
    p.add_argument('--save', default='runs/infer', help='[infer] 结果输出目录')
    p.add_argument('--no-classify', action='store_true', help='[infer] 只检测不分类')
    args, rest = p.parse_known_args(argv)
    if mode == 'infer':
        cmd_infer(args, rest)
    else:
        if rest:
            p.error(f'无法识别的参数: {rest} (子脚本参数请用 --set 步骤.参数=值)')
        cmd_train(args)


if __name__ == '__main__':
    main()
