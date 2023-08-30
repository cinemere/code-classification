import logging
import os

from src.params import *
from src.baseline.dataloader import BaselineDataset

logger = logging.getLogger(__name__)

def run_baseline(args, run_name):
    from liblinear.liblinearutil import (
        problem, parameter, train, predict, evaluations, save_model
    )

    # import data
    data = BaselineDataset(min_number_of_files_in_class=args.min_number_of_files_in_class)
    logger.info(f"{len(data.vocab)=}")
    logger.info(f"Dataset is setup!")

    if args.mode == "eval":
        logger.info(f"Baseline in {args.mode} mode.")
        
        # preprocessing    
        y_train, x_train, y_val, x_val = data.get_input_train_val(args.traintestsplit)
        logger.info(f"Data in splitted to train {len(y_train)=} and val {len(y_val)=}.")

        # training
        prob  = problem(y_train, x_train)
        param = parameter(args.liblinear_params)
        model = train(prob, param)
        logger.info(f"Model training is finished.")

        # evaluation
        predicted_labels, _, _ = predict(y_val, x_val, model)  # predicted_labels, accuracy, p_values
        logger.info(f"Validation labels are predicted.")

        (accuracy, mse, sq_corr_coef) = evaluations(y_val, predicted_labels)
        logger.info(f"{accuracy=} {mse=} {sq_corr_coef=}")
        
        # save model
        if args.save_model:
            model_file_name = os.path.join(PATH_SAVE_MODEL, run_name)
            if not os.path.exists(PATH_SAVE_MODEL): os.makedirs(PATH_SAVE_MODEL)
            save_model(model_file_name, model)

        # save predictions
        if args.save_predictions:
            predicted_folders = data.decode_predictions(predicted_labels)
            predictions_file_name = os.path.join(PATH_SAVE_PREDICTIONS, run_name)
            if not os.path.exists(PATH_SAVE_PREDICTIONS): os.makedirs(PATH_SAVE_PREDICTIONS)
            with open(f"{predictions_file_name}.txt", 'w') as f:
                for filename, folder in zip(data.filenames, predicted_folders):
                    f.write(f"{filename} --> {folder}\n")
        
        # save metrics
        if args.save_metrics:
            metrics_file_name = os.path.join(PATH_SAVE_METRICS, run_name)
            if not os.path.exists(PATH_SAVE_METRICS): os.makedirs(PATH_SAVE_METRICS)
            with open(f"{metrics_file_name}.txt", 'w') as f:
                f.write(f"{accuracy=} {mse=} {sq_corr_coef=}")

                # save metrics 2.0
                from src.word2vec.trainer import evaluate
                metrics2 = evaluate(y_val, predicted_labels, le)
                m_vals, m_arrs, m_count = metrics2

                f.write(f"\nAVERAGE METRICS:\n")
                for key, value in m_vals.items():
                    f.write(f"{key:30}={value:10.8}\n")

                f.write(f"\nBY-CLASS METRICS:\n")
                for key, values in m_arrs.items():
                    f.write(f"{key}:\n")
                    for i, value in enumerate(values):
                        classname = f"({data.decode_predictions([i + 1])[0]})"
                        # classname = f"({le.inverse_transform([i])[0]})"
                        f.write(f"{i}\t{classname:30}\t{value:10.4}\n")
                
                f.write(f"\nCOUNTS:\n")
                for i, (count_x, count_y) in enumerate(zip(m_count['counts_y_pred_x'], m_count['counts_y_pred'])):
                    classname = f"({data.decode_predictions([count_x])[0]})"
                    # classname = f"({le.inverse_transform([count_x])[0]})"
                    if count_x in m_count['counts_y_val_x']:
                        count_v = m_count['counts_y_val'][m_count['counts_y_val_x'] == count_x][0]
                    else:
                        count_v = 0
                    f.write(f"{count_x:3}\t{classname:30}\t{count_y:3} (out of {count_v:6})\n")

    elif args.mode == "predict":
        logger.info(f"Baseline in {args.mode} mode.")

        # preprocessing    
        y_train, x_train = data.get_input_data()
        logger.info(f"Input data is loaded {len(y_train)=}.")

        # training
        prob  = problem(y_train, x_train)
        param = parameter(args.liblinear_params)
        model = train(prob, param)
        logger.info(f"Model training is finished.")
        
        # prediction
        infer = BaselineDataset(mode='infer')
        _, x_infer = infer.get_input_data()

        predicted_labels, _, _ = predict([], x_infer, model)  # predicted_labels, accuracy, p_values
        
        # save model
        if args.save_model:
            model_file_name = os.path.join(PATH_SAVE_MODEL, run_name)
            if not os.path.exists(PATH_SAVE_MODEL): os.makedirs(PATH_SAVE_MODEL)
            save_model(model_file_name, model)

        # save predictions
        if args.save_predictions:
            predicted_folders = data.decode_predictions(predicted_labels)
            predictions_file_name = os.path.join(PATH_SAVE_PREDICTIONS, run_name)
            if not os.path.exists(PATH_SAVE_PREDICTIONS): os.makedirs(PATH_SAVE_PREDICTIONS)
            with open(f"{predictions_file_name}.txt", 'w') as f:
                for filename, folder in zip(infer.filenames, predicted_folders):
                    f.write(f"{filename} --> {folder}\n")