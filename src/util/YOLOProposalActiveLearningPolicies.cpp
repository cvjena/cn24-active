/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ActiveLearningPolicy.h"

#include <cmath>

namespace Conv {

datum YOLOProposalSum1vs2ActiveLearningPolicy::ScoreW(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index, std::vector<datum>& class_weights) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);

  const bool use_class_weights = (class_weights.size() == classes_);
  // LOGDEBUG << "Using class weights: " << use_class_weights;
  // LOGDEBUG << "Class weight vector length: " << class_weights.size();
  // LOGDEBUG << "Class prediction vector length: " << classes_;

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      bool has_proposal = false;
      for(unsigned int p = 0; p < proposals.size(); p++) {
        if(proposals[p].cell_id == cell_id)
          has_proposal = true;
      }
      if(!has_proposal)
        continue;


      datum max_class_score = 0;
      datum second_max_class_score = 0;

      unsigned int max_class = 0;
      unsigned int second_max_class = 0;      

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) {
          second_max_class_score = max_class_score;
          second_max_class = max_class;
          max_class_score = class_prob;
          max_class = c;
        }
      }

      datum box_score = (1.0f - (max_class_score - second_max_class_score));

      if(use_class_weights)
        // w2prop total_score += class_weights[second_max_class] * (box_score * box_score);
        // wwprop total_score += class_weights[max_class] * class_weights[second_max_class] * (box_score * box_score);
        // wprop  total_score += class_weights[max_class] * (box_score * box_score);
        total_score += class_weights[max_class] * (box_score * box_score);
      else
        total_score += (box_score * box_score);
    }
  }

  return total_score;
}

datum YOLOProposalMax1vs2ActiveLearningPolicy::ScoreW(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index, std::vector<datum>& class_weights) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);
  const bool use_class_weights = (class_weights.size() == classes_);

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      bool has_proposal = false;
      for(unsigned int p = 0; p < proposals.size(); p++) {
        if(proposals[p].cell_id == cell_id)
          has_proposal = true;
      }
      if(!has_proposal)
        continue;
      datum max_class_score = 0;
      datum second_max_class_score = 0;

      unsigned int max_class = 0;

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) {
          second_max_class_score = max_class_score;
          max_class_score = class_prob;
          max_class = c;
        }
      }

      const datum box_score = (1.0f - (max_class_score - second_max_class_score));
      datum cell_score = 0.0f;

      if(use_class_weights)
        // w2prop total_score += class_weights[second_max_class] * (box_score * box_score);
        // wwprop total_score += class_weights[max_class] * class_weights[second_max_class] * (box_score * box_score);
        // wprop
        cell_score = class_weights[max_class] * (box_score * box_score);
      else
        cell_score = (box_score * box_score);

      if(cell_score > total_score)
      total_score = cell_score;
    }
  }

  return total_score;
}

datum YOLOProposalAvg1vs2ActiveLearningPolicy::ScoreW(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index, std::vector<datum>& class_weights) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);
  const bool use_class_weights = (class_weights.size() == classes_);

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;
  datum total_score_components = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      bool has_proposal = false;
      for(unsigned int p = 0; p < proposals.size(); p++) {
        if(proposals[p].cell_id == cell_id)
          has_proposal = true;
      }
      if(!has_proposal)
        continue;
      datum max_class_score = 0;
      datum second_max_class_score = 0;

      unsigned int max_class = 0;

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) {
          second_max_class_score = max_class_score;
          max_class_score = class_prob;
          max_class = c;
        }
      }

      const datum box_score = (1.0f - (max_class_score - second_max_class_score));

      if(use_class_weights)
        // w2prop total_score += class_weights[second_max_class] * (box_score * box_score);
        // wwprop total_score += class_weights[max_class] * class_weights[second_max_class] * (box_score * box_score);
        // wprop
        total_score += class_weights[max_class] * (box_score * box_score);
      else
        total_score += (box_score * box_score);

      total_score_components += 1;
    }
  }

  if(total_score_components > 0) {
    return total_score / total_score_components;
  } else {
    return (datum)0;
  }
}

}
