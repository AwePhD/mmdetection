# ---------------------------------------------
# ----------------- EVALUATION ----------------
# ---------------------------------------------
# IMPORTANT:
# - test_evaluator is used to output pickles
# - val_evaluator is used to evaluate the pickles (not used during training)
data_root = "/dataset"
test_cfg = dict(type='TestLoop')

test_evaluator = dict(
    type='ReIDDetMetric',
    ann_file=data_root + "/annotations/test_sysu.csv",
    split_column="split_sysu",
    metric='mAP',
    metric_options=dict(
        n_samples=2900, gallery_threshold=.30, gallery_size=100))

# ---------------------------------------------
# --------------- TRANSFORMS ------------------
# ---------------------------------------------
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
    dict(type="Resize", scale=(1500, 900), keep_ratio=True),
    dict(
        type="PackReIDDetInputs",
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# ---------------------------------------------
# --------------- DATALOADERS -----------------
# ---------------------------------------------

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
test_dataloader = dict(
    shuffle=False,
    batch_size=8,  # Used only to generate pickle results.
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
