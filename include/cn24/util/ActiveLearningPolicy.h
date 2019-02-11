/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_ACTIVELEARNINGPOLICY_H
#define CONV_ACTIVELEARNINGPOLICY_H

#include "Tensor.h"
#include "SegmentSet.h"
#include <string>
#include <random>

namespace Conv {

class ActiveLearningPolicy {
public:
  virtual datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) = 0;
  virtual datum ScoreW(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index, std::vector<datum>& class_weights) {
    LOGDEBUG << "Ignoring class weights :(";
    return Score(output, metadata, index);
  }

};

class DefaultActiveLearningPolicy : public ActiveLearningPolicy {
public:
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) {
    return 1;
  }
};

class RandomActiveLearningPolicy : public ActiveLearningPolicy {
public:
  RandomActiveLearningPolicy(int RANDOM_SEED) : gen(RANDOM_SEED), dist(1,10) {}
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) {
    return dist(gen);
  }
private:
  std::mt19937_64 gen;
  std::uniform_real_distribution<datum> dist;
};

class YOLOActiveLearningPolicy : public ActiveLearningPolicy {
public:
  explicit YOLOActiveLearningPolicy(JSON yolo_configuration)
      : yolo_configuration(yolo_configuration) {
    if(yolo_configuration.count("horizontal_cells") != 1 || !yolo_configuration["horizontal_cells"].is_number()) {
      FATAL("YOLO yolo_configuration property horizontal_cells missing!");
    }
    horizontal_cells_ = yolo_configuration["horizontal_cells"];

    if(yolo_configuration.count("vertical_cells") != 1 || !yolo_configuration["vertical_cells"].is_number()) {
      FATAL("YOLO yolo_configuration property vertical_cells missing!");
    }
    vertical_cells_ = yolo_configuration["vertical_cells"];

    if(yolo_configuration.count("boxes_per_cell") != 1 || !yolo_configuration["boxes_per_cell"].is_number()) {
      FATAL("YOLO yolo_configuration property boxes_per_cell missing!");
    }
    boxes_per_cell_ = yolo_configuration["boxes_per_cell"];

    if(yolo_configuration.count("confidence_threshold") == 1 && yolo_configuration["confidence_threshold"].is_number()) {
      confidence_threshold_ = yolo_configuration["confidence_threshold"];
    }
  }
  virtual datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) = 0;
  static ActiveLearningPolicy* CreateWithName(std::string& name, JSON yolo_configuration, int RANDOM_SEED);
protected:
  JSON yolo_configuration;
  unsigned int horizontal_cells_ = 0;
  unsigned int vertical_cells_ = 0;
  unsigned int boxes_per_cell_ = 0;
  datum confidence_threshold_ = 0.2;
};

class YOLOWholeImageDiffActiveLearningPolicy : public YOLOActiveLearningPolicy {
public:
  explicit YOLOWholeImageDiffActiveLearningPolicy(JSON yolo_configuration) : YOLOActiveLearningPolicy(yolo_configuration) {};
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index);
};

class YOLOWholeImage1vs2ActiveLearningPolicy : public YOLOActiveLearningPolicy {
public:
  explicit YOLOWholeImage1vs2ActiveLearningPolicy(JSON yolo_configuration) : YOLOActiveLearningPolicy(yolo_configuration) {};
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index);
};

class YOLOProposalSum1vs2ActiveLearningPolicy : public YOLOActiveLearningPolicy {
public:
  explicit YOLOProposalSum1vs2ActiveLearningPolicy(JSON yolo_configuration) : YOLOActiveLearningPolicy(yolo_configuration) {};
  datum ScoreW(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index, std::vector<datum>& class_weights);
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) {
      std::vector<datum> empty_class_weights = {};
      return ScoreW(output, metadata, index, empty_class_weights);
  }
};

class YOLOProposalMax1vs2ActiveLearningPolicy : public YOLOActiveLearningPolicy {
public:
  explicit YOLOProposalMax1vs2ActiveLearningPolicy(JSON yolo_configuration) : YOLOActiveLearningPolicy(yolo_configuration) {};
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) {
    std::vector<datum> empty_class_weights = {};
    return ScoreW(output, metadata, index, empty_class_weights);
  }
  datum ScoreW(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index, std::vector<datum>& class_weights);
};

class YOLOProposalAvg1vs2ActiveLearningPolicy : public YOLOActiveLearningPolicy {
public:
  explicit YOLOProposalAvg1vs2ActiveLearningPolicy(JSON yolo_configuration) : YOLOActiveLearningPolicy(yolo_configuration) {};
  datum Score(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index) {
    std::vector<datum> empty_class_weights = {};
    return ScoreW(output, metadata, index, empty_class_weights);
  }
  datum ScoreW(Tensor &output, DatasetMetadataPointer* metadata, unsigned int index, std::vector<datum>& class_weights);
};
}

#endif
