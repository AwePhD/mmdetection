# ---------------------------------------------
# ----------------- EVALUATION ----------------
# ---------------------------------------------
metric_evaluator = dict(
    type='ReIDDetMetric',
    metric='mAP',
    metric_options=dict(
        n_samples=2900, gallery_threshold=.30, gallery_size=100))
metric_dump = dict(type="DumpReIDDetResults", )

# First dump the results in metric_dump
# Second use offline evaluation with metric_evaluator
test_evaluator = metric_dump
test_cfg = dict(type='TestLoop')

# ---------------------------------------------
# --------------- TRANSFORMS ------------------
# ---------------------------------------------

transfrom_normalize = dict(
    type="Normalize",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
transform_pad = dict(type="Pad", size_divisor=1)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadReIDDetAnnotations"),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (667, 400),
                        (1000, 600),
                        (1333, 800),
                        (1500, 900),
                        (1666, 1000),
                    ],
                    keep_ratio=True,
                )
            ],
        ],
    ),
    dict(type="PackReIDDetInputs"),
]

# NOTE: Original used 'MultiScaleFlipAug'. But it was useless with its config.
# So we replace with a simple list.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadReIDDetAnnotations", is_eval=True),
    # flip and flip_direction are in the default meta_keys of PackReIDDetInputs
    dict(type="RandomFlip", prob=0.0),
    dict(type="Resize", scale=(1500, 900), keep_ratio=True),
    dict(type="PackReIDDetInputs"),
]

# ---------------------------------------------
# --------------- DATALOADERS -----------------
# ---------------------------------------------

data_root = "/dataset"
dataset_type = "CUHK_SYSU"
num_workers = 2
train_dataloader = dict(
    shuffle=True,
    batch_size=8,
    num_workers=num_workers,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        filter_cfg=dict(filter_empty_gt=False),
        ann_file="annotations/train_sysu.json",
        data_root=data_root,
        pipeline=train_pipeline,
    ),
)

# See evalutation section
evaluate_batch_size = 1
dump_batch_size = 8  # Needs ~ 7GB of VRAM
test_batch_size = (
    evaluate_batch_size
    if test_evaluator["type"] == 'ReIDDetMetric'  # according to eval or dumb
    else dump_batch_size
)
test_dataloader = dict(
    shuffle=False,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        filter_cfg=dict(filter_empty_gt=False),
        ann_file="annotations/test_sysu.json",
        data_root=data_root,
        pipeline=test_pipeline,
    ),
)
