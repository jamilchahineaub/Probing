function errors = plot_matched_response(run_dir,repository_root)
T=readtable(fullfile(run_dir,'matched_python_response.csv'),'TextType','string'); ids=unique(T.target_id); errors=table();
f=figure('Visible','off'); tiledlayout(numel(ids),1);
for i=1:numel(ids)
    R=T(T.target_id==ids(i),:); nexttile; plot(R.time_s,1e3*R.matlab_x,'k-',R.time_s,1e3*R.python_x,'r--','LineWidth',1); ylabel('x (mm)'); title(ids(i),'Interpreter','none'); grid on;
    errors=[errors;table(ids(i),max(abs(R.matlab_x-R.python_x)),max(abs(R.matlab_v-R.python_v)),max(abs(R.matlab_a-R.python_a)),'VariableNames',{'target_id','max_x_error','max_v_error','max_a_error'})]; %#ok<AGROW>
end
xlabel('Time (s)'); legend('MATLAB','Python Stage 1'); [~,name,ext]=fileparts(run_dir); run_id=[name ext]; base=fullfile(run_dir,'figures',[run_id '__python_matlab_matched_response']);
set(f,'Color','w','Position',[100 100 900 600]); ax=findall(f,'Type','axes');set(ax,'Color','w','XColor','k','YColor','k','GridColor',[.65 .65 .65]);txt=findall(f,'Type','text');set(txt,'Color','k');exportgraphics(f,[base '.png'],'Resolution',180); exportgraphics(f,[base '.pdf'],'ContentType','vector');close(f);
writetable(errors,fullfile(run_dir,'python_matlab_crosscheck.csv')); copyfile([base '.png'],fullfile(repository_root,'results','figures')); copyfile([base '.pdf'],fullfile(repository_root,'results','figures'));
end
