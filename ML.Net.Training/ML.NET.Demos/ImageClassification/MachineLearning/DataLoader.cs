using ImageClassification.MachineLearning.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.MachineLearning;

public class DataLoader
{
    private readonly MLContext _mlContext;
    private readonly string _trainingFolder;

    public IDataView TrainSet { get; set; }
    public IDataView ValidationSet { get; set; }
    public IDataView TestSet { get; set; }

    public DataLoader(string trainingFolder)
    {
        _mlContext = new(111);
        _trainingFolder = trainingFolder;

        var dataProcessPipeline = BuildDataProcessingPipeline();
        LoadAndPrepareData(dataProcessPipeline);
    }

    private EstimatorChain<ImageLoadingTransformer> BuildDataProcessingPipeline()
    {
        return _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey")
                    .Append(_mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", imageFolder: _trainingFolder, inputColumnName: "ImagePath"));
    }

    private void LoadAndPrepareData(EstimatorChain<ImageLoadingTransformer> estimatorChain)
    {
        IEnumerable<ImageData> images = GetImageData();

        var imageData = _mlContext.Data.LoadFromEnumerable(images);
        var shuffledImageData = _mlContext.Data.ShuffleRows(imageData);

        var preparedData = estimatorChain.Fit(shuffledImageData).Transform(shuffledImageData);
        var trainSplit = _mlContext.Data.TrainTestSplit(data: preparedData, testFraction: 0.3);
        var validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

        TrainSet = trainSplit.TrainSet;
        ValidationSet = validationTestSplit.TrainSet;
        TestSet = validationTestSplit.TestSet;
    }

    private IEnumerable<ImageData> GetImageData()
    {
        var files = Directory.GetFiles(_trainingFolder, "*", searchOption: SearchOption.AllDirectories);

        foreach (var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                continue;

            var label = Path.GetFileName(file)[..3];

            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };
        }
    }
}
