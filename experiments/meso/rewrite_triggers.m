function [] = rewrite_triggers(sesdir)
    rawDir = fullfile(sesdir, 'raw');
    scratchDir = fullfile(sesdir, 'scratch');
    triggerFile = fullfile(rawDir, dir(fullfile(rawDir, '*.mat')).name);    
    h5File = fullfile(scratchDir, 'triggers.h5');
    
    if ~exist(scratchDir)
        mkdir(scratchDir);
    end
    if exist(h5File)
        delete(h5File);
    end
    
    mat = load(triggerFile);
    DI = mat.digitalInput;
    Time = seconds(DI.Time);
    n_timepoints = numel(Time);
    FireALL = DI.FireALL;
    LED470Signal = DI.LED470Signal;
    LED565Signal = DI.LED565Signal;
    VIS = DI.VIS;
    BIT0 = DI.BIT0;
    BIT1 = DI.BIT1;
    BIT2 = DI.BIT2;
    BIT3 = DI.BIT3;
    
    h5create(h5File, '/Time', size(Time));
    h5write(h5File, '/Time', Time);
    
    h5create(h5File, '/VIS', size(VIS));
    h5write(h5File, '/VIS', VIS);
    
    h5create(h5File, '/LED470Signal', size(LED470Signal));
    h5write(h5File, '/LED470Signal', LED470Signal);
    
    h5create(h5File, '/LED565Signal', size(LED565Signal));
    h5write(h5File, '/LED565Signal', LED565Signal);
    
    h5create(h5File, '/BIT0', size(BIT0));
    h5write(h5File, '/BIT0', BIT0);
    
    h5create(h5File, '/BIT1', size(BIT1));
    h5write(h5File, '/BIT1', BIT1);
    
    h5create(h5File, '/BIT2', size(BIT2));
    h5write(h5File, '/BIT2', BIT2);
    
    h5create(h5File, '/BIT3', size(BIT3));
    h5write(h5File, '/BIT3', BIT3);

