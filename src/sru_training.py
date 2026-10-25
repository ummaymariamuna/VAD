"""
SRU-based video anomaly detection training module.
This module provides a complete training pipeline for video anomaly detection using SRU (Simple Recurrent Unit).
It includes model definition, training, evaluation, and visualization components.
"""

import copy
import logging
import os
import json
from datetime import datetime 
from typing import Tuple, Dict, Any, Optional, List
import gc

import numpy as np 
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.preprocessing import label_binarize
from sru import SRU
from torch.nn import CrossEntropyLoss, Module, Dropout, Linear
from torch.optim import Adam,SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 

from dataset_split import group_train_test_split,source_video_ids



EMBEDDINGS_PATH = "/path/to/embeddings.npy"
LABELS_PATH = "/path/to/labels.npy"
# Set to None to split clips at random (segments and augmented copies of one video then leak into the test set). 
PATHS_PATH = "/path/to/paths.npy"
NUM_CLASSES=2
TEST_SIZE=0.2
VAL_SIZE = 0.1
BATCH_SIZE = 8
INPUT_SIZE = 2048
HIDDEN_SIZE = 1024
NUM_LAYERS=2
BIDIRECTIONAL = False
DROPOUT_PROB = 0.0
DROPOUT_LAYER_PROB = 0.2 
L2_REG_LAMBDA = 1e-5
EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
SAVE_DIR = "models"
LOG_DIR = "logs"
SHOW_PLOTS = True


class SRUModel(Module):

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super(SRUModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 2)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_classes = kwargs.get('num_classes', 2)

        self.model_kwargs = {
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes,
            'dropout_prob': kwargs.get('dropout_prob', 0.0),
            'layer_norm': kwargs.get('layer_norm', False),
            'highway_bias': kwargs.get('highway_bias', 0.0),
            'rescale': kwargs.get('rescale', True),
            'nn_rnn_compatible_return': kwargs.get('nn_rnn_compatible_return', False),
            'proj_input_to_hidden_first': kwargs.get('proj_input_to_hidden_first', False),
            'amp_recurrence_fp16': kwargs.get('amp_recurrence_fp16', False),
            'normalize_after': kwargs.get('normalize_after', False),
            'dropout_layer_prob': kwargs.get('dropout_layer_prob', 0.2),
            'l2_reg_lambda': kwargs.get('l2_reg_lambda', 1e-5),
        }
        self.sru_layers=SRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=self.model_kwargs['dropout_prob'],
            bidirectional=self.bidirectional,
            layer_norm=self.model_kwargs['layer_norm'],
            highway_bias=self.model_kwargs['highway_bias'],
            rescale=self.model_kwargs['rescale'],
            nn_rnn_compatible_return=self.model_kwargs['nn_rnn_compatible_return'],
            proj_input_to_hidden_first=self.model_kwargs['proj_input_to_hidden_first'],
            amp_recurrence_fp16=self.model_kwargs['amp_recurrence_fp16'],
            normalize_after=self.model_kwargs['normalize_after'],
        )

        self.dropout = Dropout(self.model_kwargs['dropout_layer_prob'])
        output_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.linear = Linear( in_features=output_size, out_features=self.num_classes)
        self.l2_reg_lambda = self.model_kwargs['l2_reg_lambda']

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device) -> 'SRUModel':
        """Rebuild a trained model from a checkpoint written by VADTrainer.save_model."""
        checkpoint=torch.load(checkpoint_path, map_location=device)
        model_info = checkpoint['model_info']
        model_kwargs = model_info.get('model_kwargs') or {
            key: model_info[key] for key in ('num_layers', 'bidirectional', 'num_classes') if key in model_info
        } 
        model = cls(model_info['input_size'], model_info['hidden_size'], **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device).eval()

    def forward(self, x):

        output_states, _ = self.sru_layers(x)
        if self.bidirectional:
            # The backward direction has seen the whole sequence at the first time step, not the last
            last_states = torch.cat([output_states[-1, :, :self.hidden_size],
                                     output_states[0, :, self.hidden_size:]], dim=1)
        else:
            last_states = output_states[-1]
        output = self.linear(self.dropout(last_states))
        return output 

    def l2_regularization(self):

        l2_reg = torch.tensor( 0., device=next(self.parameters()).device) 
        for param in self.parameters():
            l2_reg += torch.norm( param, p=2)
        return self.l2_reg_lambda * l2_reg 

    def get_model_info(self) -> Dict[str, Any]:

        return {
            'model_type': 'SRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes,
            'l2_reg_lambda': self.l2_reg_lambda,
            'model_kwargs': self.model_kwargs,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class VADTrainer:
    def __init__(self,
                 model: SRUModel, 
                 device: torch.device,
                 save_dir: str = 'models',
                 log_dir: str = 'logs'):

        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir,exist_ok=True)

        self.training_history={
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': []
        }

        log_filename = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig( 
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    def _split_indices(self, labels: np.ndarray, groups: Optional[List[str]], indices: np.ndarray,
                       size: float,random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        if groups is None:
            return train_test_split(indices, test_size=size, random_state=random_state, stratify=labels[indices]) 

        keep, held_out = group_train_test_split(labels[indices], [groups[i] for i in indices], size, random_state)
        return indices[keep], indices[held_out]


    def prepare_data(self,
                    embeddings_path: str,
                    labels_path: str,
                    paths_path: Optional[str] = None, 
                    test_size: float = 0.2, 
                    val_size: float = 0.1,
                    batch_size: int = 8,
                    random_state: int = 42) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        print("Loading data...")

        file_embeddings = np.load(embeddings_path)
        file_labels = np.load(labels_path).astype(np.int64)

        if len(file_embeddings) != len(file_labels):
            raise ValueError("The length of the embeddings and labels should be the same")
        if len(file_embeddings.shape) != 3:
            raise ValueError(f"The embeddings should be a 3D array [instances, sequence, features]. "
                           f"Found {len(file_embeddings.shape)}D instead.")

        if file_labels.min() < 0 or file_labels.max() >= self.model.num_classes: 
            raise ValueError(f"Labels range from {file_labels.min()} to {file_labels.max()}, but the model has "
                             f"{self.model.num_classes} classes. Set NUM_CLASSES to match the labels.")

        print("Files Loaded Successfully")
        print(f"Video Embeddings Shape: {file_embeddings.shape}")
        print(f"Video Labels Shape: {file_labels.shape}")

        if paths_path:
            groups = source_video_ids( np.load(paths_path))
            if len(groups) != len(file_labels):
                raise ValueError("The length of the video paths and labels should be the same")
            print(f"Splitting by source video ({len(set(groups))} source videos)")
        else:
            groups = None
            print("Warning: no video paths given; splitting clips at random. Clips cut from the same source video "
                  "(and their augmented copies) will appear in both train and test sets.")

        all_indices = np.arange(len(file_labels))
        train_indices, test_indices = self._split_indices(file_labels, groups, all_indices, test_size, random_state)
        val_indices = None
        if val_size > 0:
            train_indices, val_indices = self._split_indices(file_labels, groups, train_indices, val_size,
                                                             random_state)

        def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
            embeddings = torch.from_numpy(file_embeddings[indices]).float().to(self.device)
            labels = torch.from_numpy(file_labels[indices]).to(self.device)
            print(f'Shape of Embeddings: {tuple(embeddings.shape)}, Labels: {tuple(labels.shape)}')
            return DataLoader(TensorDataset(embeddings, labels), shuffle=shuffle, batch_size=batch_size)

        print('Train split:')
        train_loader=make_loader(train_indices, shuffle=True)
        val_loader = None
        if val_indices is not None:
            print('Validation split:')
            val_loader = make_loader(val_indices,shuffle=False)
        print('Test split:')
        test_loader = make_loader(test_indices, shuffle=False)

        del file_embeddings,file_labels
        gc.collect() 


        return train_loader, val_loader, test_loader

    def compute_accuracy(self, data_loader: DataLoader) -> float:

        self.model.eval()
        correct = 0
        total = 0


        with torch.no_grad():
            for videos, labels in data_loader:
                outputs = self.model(videos.permute(1, 0, 2))
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)


        return 100 * correct / total

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 50, 
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              save_best_model: bool = True) -> Dict[str, List[float]]:

        criterion = CrossEntropyLoss()

        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learning_rate) 
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_type}, Learning Rate: {learning_rate}")

        best_val_accuracy = -1.0
        best_state = None

        for epoch in range( epochs): 
            self.model.train() 
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            epoch_start_time = datetime.now()

            progress_bar = tqdm(
                enumerate(  train_loader), 
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=len(train_loader)
            )


            for i, (videos, labels) in progress_bar:
                videos=videos.permute(1, 0, 2)

                outputs = self.model(videos)
                labels = labels.long()
                loss = criterion(outputs, labels) + self.model.l2_regularization()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                batch_accuracy = 100 * total_correct / total_samples
                progress_bar.set_postfix(loss = loss.item(), accuracy=batch_accuracy)

            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100 * total_correct / total_samples

            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['train_accuracy'].append(epoch_accuracy)
            self.training_history['epoch_times'].append(epoch_time)

            log_message = f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%'


            if val_loader is not None:
                val_accuracy=self.compute_accuracy(val_loader)
                self.training_history['val_accuracy'].append(val_accuracy)
                log_message += f', Val Accuracy: {val_accuracy:.2f}%'

                # Select the best epoch on held-out data, not on training accuracy
                if save_best_model and val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_state = copy.deepcopy(self.model.state_dict())
                    self.save_model('best_model.pth', epoch + 1, val_accuracy)


            log_message += f', Time: {epoch_time:.2f}s'
            print(  log_message)
            self.logger.info(log_message)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f'Training completed. Restored best model (validation accuracy {best_val_accuracy:.2f}%)')
        else:
            print('Training completed. Using the final epoch (no validation set)')
        return self.training_history


    def evaluate(self, test_loader: DataLoader, show_plots: bool = True) -> Dict[str, Any]: 

        print("Evaluating model...")
        self.model.eval()

        all_predictions = []
        all_labels=[]
        all_probs=[]
        with torch.no_grad():
            correct = 0
            total = 0
            for videos, labels in tqdm(test_loader, desc="Evaluating"):
                videos = videos.permute(1, 0, 2)

                outputs = self.model( videos)

                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim = 1).detach().cpu().numpy()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels) 
        all_probs = np.array( all_probs)
        num_classes = self.model.num_classes

        conf_matrix = confusion_matrix(all_labels, all_predictions, labels = list(range(num_classes)))

        if num_classes == 2:
            scores=all_probs[:, 1]
            fpr, tpr, _ = roc_curve(all_labels, scores)
            precision, recall, _ = precision_recall_curve(all_labels, scores)
            average_precision = average_precision_score(all_labels, scores) 
        else: 
            one_hot=label_binarize(all_labels, classes = list(range(num_classes)))
            fpr, tpr, _ = roc_curve(one_hot.ravel(), all_probs.ravel())
            precision, recall, _ = precision_recall_curve(one_hot.ravel(), all_probs.ravel())
            average_precision = average_precision_score(one_hot, all_probs, average='micro')
        roc_auc = auc( fpr, tpr)

        results = {
            'test_accuracy': test_accuracy,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }

        self.plot_evaluation_results(results,show_plots)

        return results

    def plot_evaluation_results(self, results: Dict[str, Any], show: bool = True):

        fig,axes = plt.subplots(1, 3, figsize=(15, 5))


        sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0,1.0])
        axes[1].set_ylim([0.0, 1.05]) 
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")

        axes[2].step(results['recall'], results['precision'], color='b', alpha=0.2, where='post')
        axes[2].fill_between(results['recall'], results['precision'], step='post', alpha=0.2, color='b')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_ylim([0.0, 1.05]) 
        axes[2].set_xlim([0.0,1.0])
        axes[2].set_title(f'PR Curve (AP={results["average_precision"]:.2f})')
        plt.tight_layout()

        plot_path = os.path.join(self.save_dir, f'evaluation_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {plot_path}")


        if show:
            plt.show()
        plt.close(fig)
    def plot_training_history(self, show: bool = True):


        fig, axes=plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(self.training_history['train_loss']) + 1)

        axes[0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, self.training_history['train_accuracy'], 'r-', label='Training Accuracy')
        if self.training_history['val_accuracy']:
            axes[1].plot(epochs, self.training_history['val_accuracy'], 'g-', label='Validation Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout( )

        # Save before showing: closing the shown window discards the figure
        plot_path = os.path.join(self.save_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_path}")

        if show:
            plt.show( )
        plt.close(fig)

    def save_model(self, filename: str, epoch: int, accuracy: float):

        checkpoint={
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'model_info': self.model.get_model_info(),
            'training_history': self.training_history
        }

        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:

        checkpoint = torch.load(checkpoint_path,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%")

        return checkpoint

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device selected: CUDA")
        os.system("nvidia-smi")
        print(f"Device capability: {torch.cuda.get_device_capability(device)}")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")

    return device


def main():
    device=setup_device()

    model=SRUModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        num_classes=NUM_CLASSES,
        dropout_prob=DROPOUT_PROB,
        dropout_layer_prob=DROPOUT_LAYER_PROB,
        l2_reg_lambda = L2_REG_LAMBDA
    )

    print("\nModel Architecture:")
    print( json.dumps(model.get_model_info(), indent = 2))

    trainer=VADTrainer(model, device, SAVE_DIR, LOG_DIR)

    train_loader, val_loader, test_loader = trainer.prepare_data(
        EMBEDDINGS_PATH,
        LABELS_PATH,
        PATHS_PATH, 
        TEST_SIZE,
        VAL_SIZE,
        BATCH_SIZE 
    )
    history = trainer.train(
        train_loader,
        val_loader,
        EPOCHS,
        LEARNING_RATE,
        OPTIMIZER
    )

    trainer.plot_training_history(SHOW_PLOTS)

    results = trainer.evaluate(test_loader, SHOW_PLOTS)
    trainer.save_model(
        f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
        EPOCHS,
        results['test_accuracy']
    )

    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()