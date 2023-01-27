import paramanager as pm

ARCHITECTURE_PARAMS = [
    pm.ProtoParameter("nc_a", 3, int),
    pm.ProtoParameter("nc_b", 3, int),
    pm.ProtoParameter("ngf", 64, int),
    pm.ProtoParameter("n_down", 2, int),
    pm.ProtoParameter("n_res", 6, int),
    pm.ProtoParameter("ndf", 64, int),
]

EVALUATION_PARAMS = [
    pm.ProtoParameter("eval_ext_a", "testA", str, ["eval_a"]),
    pm.ProtoParameter("eval_ext_b", "testB", str, ["eval_b"])
]

TRAINING_PARAMS = [
    pm.ProtoParameter("lr", 0.0002),
    pm.ProtoParameter("lr_end", None),
    pm.ProtoParameter("beta1", 0.5),
    pm.ProtoParameter("beta2", 0.99),
    pm.ProtoParameter("batch_size", 1, int, ["bs"]),
    pm.ProtoParameter("epochs", 10, int, ["e"]),
    pm.ProtoParameter("lambda_cyc", 10.0, pseudonyms=["lc"]),
    pm.ProtoParameter("lambda_idt", 0.0, pseudonyms=["li"]),
    pm.ProtoParameter("pool_size", 50, int, ["p"]),
    pm.ProtoParameter("train_ext_a", "trainA", str, ["train_a"]),
    pm.ProtoParameter("train_ext_b", "trainB", str, ["train_b"]),
    pm.ProtoParameter("des_rel_lr", 0.25, float),
]
