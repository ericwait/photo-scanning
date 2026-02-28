function img = scanV600(dpi)
    % SCANV600 Scans a 48-bit raw TIFF from Epson V600 via WSL
    %
    % Usage: img = scanV600(300);
    %
    % This function automatically finds the 'interpreter' device identifier
    % to ensure it uses the correct Epson plugin driver.

    % --- 1. Settings ---
    winPath = fullfile(tempdir, 'scan_raw.tif');
    % Convert Windows temp path to WSL path (e.g. C:\Temp -> /mnt/c/Temp)
    wslPath = strrep(winPath, '\', '/');
    wslPath = strrep(wslPath, 'C:', '/mnt/c'); 
    % (Adjust drive letter logic if your temp is not on C:)

    % --- 2. Auto-Detect Device ID ---
    fprintf('Searching for Epson V600...\n');
    [status, listOut] = system('wsl scanimage -L');
    
    if status ~= 0
        error('WSL failed to run. Is wsl installed?');
    end

    % Parse the output for the 'interpreter' line
    % We look for the pattern `device `(epkowa:interpreter:.*)'`
    tokens = regexp(listOut, 'device `(epkowa:interpreter:[^'']+)''', 'tokens');
    
    if isempty(tokens)
        error('Scanner not found! Try resetting USB:\n  1. usbipd detach --busid <ID>\n  2. usbipd attach --wsl --busid <ID>');
    end
    
    deviceID = tokens{1}{1};
    fprintf('Found device: %s\n', deviceID);

    % --- 3. Construct Scan Command ---
    % --device  : Uses the specific ID we found
    % --mode    : "Color" is the standard mode
    % --depth   : 16 forces 16-bit per channel (48-bit total)
    % --format  : tiff is required to save the 16-bit data correctly
    cmd = sprintf('wsl scanimage --device "%s" --mode Color --depth 16 --resolution %d --format=tiff > "%s"', ...
                  deviceID, dpi, wslPath);

    % --- 4. Execute Scan ---
    fprintf('Scanning at %d DPI (48-bit)... this may take a moment.\n', dpi);
    [status, cmdout] = system(cmd);

    % --- 5. Import and Display ---
    if status == 0
        if exist(winPath, 'file')
            img = imread(winPath);
            
            % Verify the result
            info = imfinfo(winPath);
            fprintf('------------------------------------------\n');
            fprintf('Scan Complete!\n');
            fprintf('Dimensions: %d x %d\n', info.Width, info.Height);
            fprintf('Bit Depth:  %d (Target: 48)\n', info.BitDepth);
            fprintf('File Size:  %.2f MB\n', info.FileSize / 1024 / 1024);
            fprintf('------------------------------------------\n');
            
            imshow(img);
            title(['Epson V600 Scan: ' num2str(dpi) ' DPI']);
            
            % Optional: Clean up temp file
            % delete(winPath);
        else
            error('Scan finished but output file is missing.');
        end
    else
        error('Scan failed during execution. SANE Output:\n%s', cmdout);
    end
end
