﻿using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class RandomForestTrainer : TrainerBase<FastForestBinaryModelParameters>
    {
        public RandomForestTrainer(int numberOfLeaves, int numberOfTrees) : base()
        {
            Name = $"Random Forest-{numberOfLeaves}-{numberOfTrees}";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .FastForest(numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees);
        }
    }
}
