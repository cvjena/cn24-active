/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ActiveLearningPolicy.h"

#include <cmath>

namespace Conv {

ActiveLearningPolicy* YOLOActiveLearningPolicy::CreateWithName(std::string &name, JSON yolo_configuration, int RANDOM_SEED) {
  if(name.compare("wholeimagediff") == 0) {
    return new YOLOWholeImageDiffActiveLearningPolicy(yolo_configuration);
  } else if(name.compare("wholeimage1vs2") == 0) {
    return new YOLOWholeImage1vs2ActiveLearningPolicy(yolo_configuration);
  } else if(name.compare("proposalsum1vs2") == 0) {
    return new YOLOProposalSum1vs2ActiveLearningPolicy(yolo_configuration);
  } else if(name.compare("proposalmax1vs2") == 0) {
    return new YOLOProposalMax1vs2ActiveLearningPolicy(yolo_configuration);
  } else if(name.compare("proposalavg1vs2") == 0) {
    return new YOLOProposalAvg1vs2ActiveLearningPolicy(yolo_configuration);
  } else if(name.compare("random") == 0) {
    return new RandomActiveLearningPolicy(RANDOM_SEED);
  } else {
    LOGERROR << "Unknown active learning policy: " << name;
    return new DefaultActiveLearningPolicy();
  }
}

datum YOLOWholeImageDiffActiveLearningPolicy::Score(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      datum max_class_score = 0;

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) max_class_score = class_prob;
      }

      datum cell_score = 0;
      datum max_iou = 0;

      // Loop over all possible boxes
      for (unsigned int b = 0; b < boxes_per_cell_; b++) {
        // Get predicted IOU
        const datum iou = output.data_ptr_const()[iou_index + (cell_id * boxes_per_cell_ + b) * 5 + 4];
        if(iou > max_iou)
          max_iou = iou;
      }

      const datum box_score = max_iou - max_class_score;
      cell_score += (box_score * box_score);
      total_score += cell_score;
    }
  }

  return total_score;
}

datum YOLOWholeImage1vs2ActiveLearningPolicy::Score(Tensor &output, DatasetMetadataPointer *metadata, unsigned int index) {
#ifdef BUILD_OPENCL
  output.MoveToCPU();
#endif
  DetectionMetadataPointer* detection_metadata = (DetectionMetadataPointer*) metadata;
  DetectionMetadata& proposals = *(detection_metadata[index]);

  unsigned int total_maps = output.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  unsigned int classes_ = maps_per_cell - (5 * boxes_per_cell_);

  // Prepare indices into the prediction array
  unsigned int sample_index = index * output.maps();
  unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
  unsigned int iou_index = sample_index;

  datum total_score = 0;

  // Loop over all cells
  for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
    for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
      unsigned int cell_id = vcell * horizontal_cells_ + hcell;

      datum max_class_score = 0;
      datum second_max_class_score = 0;

      // Loop over all classes
      for (unsigned int c = 0; c < classes_; c++) {
        datum class_prob = output.data_ptr_const()[class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id];
        if(class_prob > max_class_score) {
          second_max_class_score = max_class_score;
          max_class_score = class_prob;
        }
      }

      datum cell_score = 0;
      datum max_iou = 0;

      // Loop over all possible boxes
      for (unsigned int b = 0; b < boxes_per_cell_; b++) {
        // Get predicted IOU
        const datum iou = output.data_ptr_const()[iou_index + (cell_id * boxes_per_cell_ + b) * 5 + 4];
        if(iou > max_iou)
          max_iou = iou;
      }

      const datum box_score = max_iou * (1.0f - (max_class_score - second_max_class_score));
      cell_score += (box_score * box_score);
      total_score += cell_score;
    }
  }

  return total_score;
}

}