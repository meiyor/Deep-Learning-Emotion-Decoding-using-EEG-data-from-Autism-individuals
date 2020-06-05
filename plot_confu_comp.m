function plot_confu_comp(C,ll,pos_max)
%% moving across the folds and subjects presented in test sets plese be sure all the folds have the same number of instances
%% if not proceed to modify this code to make it more balanced.

labh=pos_max.*ones([1 4]);
imagesc(C./labh);
colormap(jet)

textStrings = num2str(C(:),'%i');  %# Create strings from the matrix values
textStringsP = num2str(mean((C(:)./labh)')','%0.2f');  %# Create strings from the matrix values
%textStrings = num2str(C(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
textStringsP = strtrim(cellstr(textStringsP));  %# Remove any space padding
[x,y] = meshgrid(1:4);   %# Create x and y coordinates for the strings
for p=1:length(textStrings)
 textStrings{p}=['(' textStrings{p} ')'];
end;
hStrings = text(x(:),y(:)+0.07,textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center','FontSize',16);
hStringsP = text(x(:),y(:)-0.07,textStringsP(:),...      %# Plot the strings
                'HorizontalAlignment','center','FontSize',16);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(C(:) > midValue,1,3);
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
set(hStringsP,{'Color'},num2cell(textColors,2));  %# Change the text colors

colorbar
set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',ll,...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',ll,...
        'TickLength',[0 0],'FontSize',35);
xtickangle(35);
