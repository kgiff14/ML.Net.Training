using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MultiClassClassification.MachineLearning.DataModels;

namespace MultiClassClassification.MachineLearning.Common
{
    public class TrainerBase<TParamters> : ITrainerBase
        where TParamters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "MultiClassClassification.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<MulticlassPredictionTransformer<TParamters>, TParamters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(111);
        }

        public MulticlassClassificationMetrics Evalute()
        {
            var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);

            return MlContext.MulticlassClassification.Evaluate(testSetTransform);
        }

        public void Fit(string fileName)
        {
            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException(fileName);
            }

            DataSplit = LoadAndPrepareData(fileName);
            var pipeline = BuildDataProcessingPipeline();
            var trainingPipeline = pipeline.Append(Model).Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            TrainedModel = trainingPipeline.Fit(DataSplit.TrainSet);
        }

        public void Save()
        {
            MlContext.Model.Save(TrainedModel, DataSplit.TrainSet.Schema, ModelPath);
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<PenguinData>(trainingFileName, hasHeader: true, separatorChar: ',');

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(PenguinData.Label), outputColumnName: "Label")
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Island", outputColumnName: "IslandFeaturized"))
                .Append(MlContext.Transforms.Concatenate("Features",
                                                         "IslandFeaturized",
                                                         nameof(PenguinData.CulmenLength),
                                                         nameof(PenguinData.CulmenDepth),
                                                         nameof(PenguinData.BodyMass),
                                                         nameof(PenguinData.FilperLength),
                                                         "SexFeaturized"))
                .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
