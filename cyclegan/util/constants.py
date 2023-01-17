import paramanager as pm

ARCHITECTURE_PARAMS = [
    pm.ProtoParameter("nc_a", 3, int),
    pm.ProtoParameter("nc_b", 3, int),
    pm.ProtoParameter("ngf", 64, int),
    pm.ProtoParameter("n_down", 2, int),
    pm.ProtoParameter("n_res", 6, int),
    pm.ProtoParameter("ndf", 64, int),
]

TRAINING_PARAMS = [
    pm.ProtoParameter("lr", 0.0002),
    pm.ProtoParameter("beta1", 0.5),
    pm.ProtoParameter("beta2", 0.99),
    pm.ProtoParameter("batch_size", 1, int, ["bs"]),
    pm.ProtoParameter("epochs", 10, int, ["e"]),
    pm.ProtoParameter("lambda_cyc", 1.0, pseudonyms=["lc"]),
    pm.ProtoParameter("lambda_idt", 0.0, pseudonyms=["li"])
]
