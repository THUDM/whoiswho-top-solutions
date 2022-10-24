try:
    from .rw_w2v_cluster import RwW2vClusterMaster
except Exception:
    from rw_w2v_cluster import RwW2vClusterMaster
import os


if __name__ == '__main__':
    rw_conf = {
        'repeat_num': 10,
        'num_walk': 5,
        'walk_len': 20,
        'rw_dim': 100,
        'neg': 25,
        'window': 10
    }
    rule_conf = {
        'w_author': 1.5,
        'w_venue': 1.0,
        'w_org': 1.0,
        'w_word': 0.33
    }
    w2v_model = 'tvt'
    text_weight = 1
    db_eps = 0.2
    db_min = 4
    mode = 'test'

    # 改成自己的数据路径
    # data_base = os.path.join(os.environ['DATABASE'], 'WhoIsWho')
    data_base = "../data"
    
    rw_master = RwW2vClusterMaster(rw_conf=rw_conf, rule_conf=rule_conf, mode=mode,
                                   w2v_model=w2v_model, text_weight=text_weight,
                                   db_eps=db_eps, db_min=db_min,
                                   comment='training set',
                                   text_processed=False,
                                   idf_file='', add_abstract=False, local_idf_pc=False,
                                   local_idf_ab=False, if_local_idf=False, base=data_base)
    rw_master.run()
