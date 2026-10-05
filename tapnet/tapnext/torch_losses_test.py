# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Regression coverage for masked TAPNext coordinate cross entropy."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from tapnet.tapnext import torch_losses


def reference_loss(logits, labels, mask, bins):
  indices = (labels.float() - 0.5).clamp(0, bins - 1).round().long()
  loss = torch.zeros_like(indices[..., 0], dtype=torch.float32)
  for axis in range(2):
    scores = logits[..., axis * bins : (axis + 1) * bins].float()
    selected = scores.gather(-1, indices[..., axis : axis + 1]).squeeze(-1)
    loss = loss + torch.logsumexp(scores, -1) - selected
  weights = mask.reshape(loss.shape).float()
  return (loss * weights).sum() / weights.sum()


class CoordinateSoftmaxMaskTest(parameterized.TestCase):

  @parameterized.product(
      shape=[(1, 1, 2), (2, 3, 4), (2, 3)], singleton_mask=[False, True]
  )
  def test_values_and_gradients_match_weighted_per_point_reference(
      self, shape, singleton_mask
  ):
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(*shape, 8, generator=generator, requires_grad=True)
    labels = torch.rand(*shape, 2, generator=generator) * 4
    weights = (torch.arange(labels[..., 0].numel()) % 3).reshape(
        shape
    ).float() / 2
    weights.reshape(-1)[0] = 1
    mask = weights[..., None] if singleton_mask else weights
    expected = reference_loss(logits, labels, mask, bins=4)
    actual = torch_losses.coordinate_softmax(
        logits, labels.clone(), mask, pixel_size=4
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(
        actual_grad[weights == 0], torch.zeros_like(actual_grad[weights == 0])
    )

  def test_changing_masked_logits_leaves_loss_and_visible_gradients_unchanged(
      self,
  ):
    logits = torch.tensor(
        [[[[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 4.0]]]], requires_grad=True
    )
    labels = torch.full((1, 1, 2, 2), 0.5)
    mask = torch.tensor([[[[1.0], [0.0]]]])
    changed = logits.detach().clone()
    changed[..., 1, :] = torch.tensor([-100.0, 100.0, -200.0, 200.0])
    changed.requires_grad_()
    first = torch_losses.coordinate_softmax(logits, labels.clone(), mask, 2)
    second = torch_losses.coordinate_softmax(changed, labels.clone(), mask, 2)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        torch.autograd.grad(first, logits)[0],
        torch.autograd.grad(second, changed)[0],
    )

  @parameterized.parameters(False, True)
  def test_all_visible_mask_matches_unweighted_cross_entropy(self, singleton):
    logits = torch.arange(48.0).reshape(2, 3, 8) / 10
    labels = torch.full((2, 3, 2), 1.5)
    mask = torch.ones((2, 3, 1) if singleton else (2, 3))
    actual = torch_losses.coordinate_softmax(logits, labels.clone(), mask, 4)
    torch.testing.assert_close(actual, reference_loss(logits, labels, mask, 4))

  def test_public_backward_helper_does_not_train_occluded_coordinate_logits(
      self,
  ):
    class SmallTracker(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.coords = torch.nn.Parameter(torch.ones((1, 1, 2, 2)))
        self.logits = torch.nn.Parameter(
            torch.tensor([[[[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 4.0]]]])
        )
        self.visible = torch.nn.Parameter(torch.zeros((1, 1, 2, 1)))

      def forward(self, video, query_points):
        del video, query_points
        return self.coords, self.logits, self.visible, None

    model = SmallTracker()
    batch = {
        "video": torch.zeros((1, 1, 2, 2, 3)),
        "query_points": torch.zeros((1, 2, 3)),
        "target_points": torch.full((1, 2, 1, 2), 0.5),
        "visible": torch.tensor([[[[1.0]], [[0.0]]]]),
    }
    value = torch_losses.tapnext_loss_and_grad(model, batch)
    self.assertGreater(value, 0)
    self.assertTrue(torch.isfinite(model.logits.grad).all())
    torch.testing.assert_close(
        model.logits.grad[..., 1, :],
        torch.zeros_like(model.logits.grad[..., 1, :]),
    )
    self.assertGreater(
        torch.count_nonzero(model.logits.grad[..., 0, :]).item(), 0
    )


if __name__ == "__main__":
  absltest.main()
