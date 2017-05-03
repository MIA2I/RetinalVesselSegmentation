num = {'01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20'};

SeAvg = 0;
SpAvg = 0;
PrAvg = 0;
F1Avg = 0;
GAvg = 0;
MCCAvg = 0;
AccAvg = 0;

T = 0.85;
iteration = 50000;
manualID = 1;
file = fopen(strcat('./', int2str(iteration), '/QuantitativeResults1.txt'),'w');

for index = 1:20
    
    label = imread(strcat('./', int2str(iteration), '/', num{index}, '_label.png'));
    gt = imread(strcat('./', int2str(iteration), '/', num{index}, '_manual', int2str(manualID), '.gif'));
    mask = imread(strcat('./', int2str(iteration), '/', num{index},'_test_mask.gif'));

    pred = label;
    pred(pred<T*255)=0;
    pred(pred>0)=1;

    [ Se, Sp, Precision, F1, G, MCC, Acc ] = Accuracy(pred, gt, mask);
    imwrite(pred*255, strcat('./', int2str(iteration), '/', num{index}, '_prediction.png'));
    
    result=sprintf('No. %d: Se = %.6g, Sp = %.6g, Pr = %.6g, F1 = %.6g, G = %.6g, MCC = %.6g, Acc = %.6g\r\n', index, Se, Sp, Precision, F1, G, MCC, Acc);
    fprintf(file, result);
    
    [height, width, rim] = size(mask);
    scores = double(label)./255.0;
    scores = reshape(scores, [1, height*width]);
    labels = reshape(gt, [1, height*width]);
    
    SeAvg = SeAvg + Se / 100.0;
    SpAvg = SpAvg + Sp / 100.0;
    PrAvg = PrAvg + Precision / 100.0;
    F1Avg = F1Avg + F1 / 100.0;
    GAvg = GAvg + G / 100.0;
    MCCAvg = MCCAvg + MCC / 100.0;
    AccAvg = AccAvg + Acc / 100.0;
    
end

SeAvg = SeAvg / index;
SpAvg = SpAvg / index;
PrAvg = PrAvg / index;
F1Avg = F1Avg / index;
GAvg = GAvg / index;
MCCAvg = MCCAvg / index;
AccAvg = AccAvg / index;

result=sprintf('Average: Se = %.6g, Sp = %.6g, Pr = %.6g, F1 = %.6g, G = %.6g, MCC = %.6g, Acc = %.6g\r\n', SeAvg, SpAvg, PrAvg, F1Avg, GAvg, MCCAvg, AccAvg);
disp(result);
fprintf(file, result);
fclose(file);
