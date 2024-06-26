import numpy as np
from tensorflow import keras
import tensorflow as tf


class Defensive_Distiller_heatmap(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        # USE HEATMAP
        if len(data) == 3:
            x_pre, y, sample_weight = data
        else:
            x_pre, y = data

        x_pre = np.swapaxes(x_pre, 0, 1)
        img = x_pre[0]
        heatmap = x_pre[1]

        # Forward pass of teacher
        teacher_predictions = self.teacher(img, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(heatmap, training=True)

            # Compute losses
            if len(data) == 3:
                student_loss = self.student_loss_fn(teacher_predictions, student_predictions, sample_weight=sample_weight)
            else:
                student_loss = self.student_loss_fn(teacher_predictions, student_predictions)  # TODO soft

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            if len(data) == 3:

                distillation_loss = (
                        self.distillation_loss_fn(
                            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                            tf.nn.softmax(student_predictions / self.temperature, axis=1), sample_weight=sample_weight
                        )
                        * self.temperature ** 2
                )
            else:
                distillation_loss = (
                        self.distillation_loss_fn(
                            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                            tf.nn.softmax(student_predictions / self.temperature, axis=1),

                        )
                        * self.temperature ** 2
                )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        # x, y = data
        x_pre, y = data
        x_pre = np.swapaxes(x_pre, 0, 1)
        # y = y[0]
        img = x_pre[0]
        heatmap = x_pre[1]
        # Compute predictions
        y_prediction_t = self.teacher(img, training=False)

        y_prediction = self.student(heatmap, training=False)
        # TODO soft
        # Calculate the loss
        student_loss = self.student_loss_fn(y_prediction_t, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def call(self, data):
        x_pre = np.swapaxes(data, 0, 1)
        # y = y[0]
        img = x_pre[0]
        heatmap = x_pre[1]
        return self.student(heatmap, training=False)
# TODO
