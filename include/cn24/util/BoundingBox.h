/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file BoundingBox.h
 * @class BoundingBox
 * @brief Support class for bounding boxes
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_BOUNDINGBOX_H
#define CONV_BOUNDINGBOX_H

#include "Config.h"

namespace Conv {

class YOLODetectionLayer;
class YOLOProposalSum1vs2ActiveLearningPolicy;
class YOLOProposalMax1vs2ActiveLearningPolicy;

class BoundingBox {
  friend class YOLODetectionLayer;
  friend class YOLOProposalSum1vs2ActiveLearningPolicy;
  friend class YOLOProposalMax1vs2ActiveLearningPolicy;
  friend class YOLOProposalAvg1vs2ActiveLearningPolicy;
public:
  BoundingBox();
  BoundingBox(datum x, datum y, datum w, datum h) : x(x), y(y), w(w), h(h) {};

  datum IntersectionOverUnion(BoundingBox* bounding_box);

  static datum Overlap1D(datum center1, datum size1, datum center2, datum size2);
  datum Intersection(BoundingBox* bounding_box);
  datum Union(BoundingBox* bounding_box);

  static bool CompareScore(const BoundingBox& box1, const BoundingBox& box2);

  datum x = 0, y = 0, w = 0, h = 0, score = 0;

  unsigned int c = 0;

  // Flag 1 is used for calculations, always set it to false after using it
  // Flag 2 is used by certain datasets, don't ever change it!
  bool flag1 = false, flag2 = false;

  bool unknown = false;

private:
  unsigned int cell_id;
};

}

#endif
