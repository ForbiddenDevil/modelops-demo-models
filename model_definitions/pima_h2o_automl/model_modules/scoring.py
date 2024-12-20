from teradataml import copy_to_sql, get_context, DataFrame, H2OPredict
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    store_byom_tmp,
    ModelContext
)


def check_java():
    import os
    import jdk
    user_home_dir = os.path.expanduser('~')
    jupyterlab_root_dir = os.path.join(user_home_dir, 'JupyterLabRoot')
    java_base_dir = jupyterlab_root_dir if os.path.isdir(jupyterlab_root_dir) else user_home_dir
    java_install_dir = os.path.join(java_base_dir, '.jdk')

    # Check if Java is already installed
    if os.path.isdir(java_install_dir) and os.listdir(java_install_dir):
        java_home_path = os.path.join(java_install_dir, os.listdir(java_install_dir)[0])
        os.environ['JAVA_HOME'] = java_home_path
        print(f"Java is installed at {java_home_path}")
    else:
        print('Installing Java...')
        jdk.install('17', path=java_install_dir)

        # Update JAVA_HOME after installation
        java_home_path = os.path.join(java_install_dir, os.listdir(java_install_dir)[0])
        os.environ['JAVA_HOME'] = java_home_path
        os.environ['PATH'] = f"{os.environ.get('PATH')}:{os.path.join(java_home_path, 'bin')}"
        print(f"Java installed at {java_home_path}")


def score(context: ModelContext, **kwargs):

    aoa_create_context()

    with open(f"{context.artifact_input_path}/model.h2o", "rb") as f:
        model_bytes = f.read()

    model = store_byom_tmp(get_context(), "byom_models_tmp", context.model_version, model_bytes)

    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    byom_target_sql = "CAST(prediction AS INT)"

    check_java()

    print("Scoring")
    h2o = H2OPredict(
        modeldata=model,
        newdata=DataFrame.from_query(context.dataset_info.sql),
        accumulate=context.dataset_info.entity_key)

    print("Finished Scoring")


    # store the predictions
    predictions_df = h2o.result
    
    # add job_id column so we know which execution this is from if appended to predictions table
    predictions_df = predictions_df.assign(job_id=context.job_id)
    cols = {}
    cols[target_name] = predictions_df['prediction']
    predictions_df = predictions_df.assign(**cols)
    predictions_df = predictions_df[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(df=predictions_df,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append")

    print("Saved predictions in Teradata")
    
    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                         predicted_df=predictions_df,
                         context=context)
