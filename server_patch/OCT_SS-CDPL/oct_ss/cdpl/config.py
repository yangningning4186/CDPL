def add_cdpl_config(cfg):
    cfg.set_new_allowed(True)
    cfg.SEMISUPNET.CDPL_ENABLED = False
    cfg.SEMISUPNET.CDPL_TAU_BASE = 0.7
    cfg.SEMISUPNET.CDPL_ALPHA_TAIL = 0.15
    cfg.SEMISUPNET.CDPL_MIN_CLS_THRESHOLD = 0.5
    cfg.SEMISUPNET.CDPL_BOOST_KEPT_SCORES = True
    cfg.set_new_allowed(False)
