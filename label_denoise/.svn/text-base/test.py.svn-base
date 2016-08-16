'''
Created on Jul 2, 2013

@author: wangxl
'''
import optparse;
import sys;
sys.path.append('.');
import random;
import utility;
import vw_feature_extract;
import data_process;
import simulate_annotation;

def mapGetVerticalSlot():
    vSet = set();
    for ln in sys.stdin: 
        (fv, eventLst, id, ts, nquery) = vw_feature_extract.getRawFea(ln);
        for e in eventLst:
            v = e['v'];
            s = int(utility.parseNumVal(e['s'].replace('s', '').strip()));
            if(v.startswith('v')): v += str(s);
            vSet.add(v);
    print(str(vSet));
    return;

def reduceGetVerticalSlot():
    vSet = set();
    for ln in sys.stdin:
        partVSet = eval(ln.strip());
        vSet.update(partVSet);
    for v in sorted(vSet): print v;
    return;

def mapGetVerticalSlotPost():
    vSet = set();
    for ln in sys.stdin: vSet.add(ln.strip());
    print(str(vSet));
    return;

def reduceGetVerticalSlotPost():
    vSet = set();
    for ln in sys.stdin:
        partVSet = eval(ln.strip());
        vSet.update(partVSet);
    for v in sorted(vSet): print v;
    return;

if __name__ == '__main__':
    map = data_process.collectMTAnnotationMap;
    reduce = data_process.collectMTAnnotationReduce;
    optParser = optparse.OptionParser("");
    optParser.add_option('-m', '--m', action='store_true', dest='m', default=False);
    optParser.add_option('-r', '--r', action='store_true', dest='r', default=False);
    optParser.add_option('--get_editor_annotation', action='store_const',
                         const='get_editor_annotation', dest='task');
    optParser.add_option('--get_mt_annotation', action='store_const',
                         const='get_mt_annotation', dest='task');
    optParser.add_option('--get_ver_dist', action='store_const',
                         const='get_ver_dist', dest='task');
    optParser.add_option('--verDistDebug', action='store_const',
                         const='verDistDebug', dest='task');
    optParser.add_option('--debug', action='store_const',
                         const='debug', dest='task');  
    optParser.add_option('--get_nquery', action='store_const',
                         const='get_nquery', dest='task');                     
    optParser.add_option('--avg_fv', action='store_const',
                         const='avg_fv', dest='task');         
    optParser.add_option('--sim_mt_merge_query', action='store_const',
                         const='sim_mt_merge_query', dest='task');         
    optParser.add_option('--extract_nquery_vertical', action='store_const',
                         const='extract_nquery_vertical', dest='task');         
    optParser.add_option('--accum_nquery_vertical', action='store_const',
                         const='accum_nquery_vertical', dest='task');
    optParser.add_option('--get_editor_uniq_nquery', action='store_const',
                         const='get_editor_uniq_nquery', dest='task');
    optParser.add_option('--get_trans_with_vertical', action='store_const',
                         const='get_trans_with_vertical', dest='task');
    optParser.add_option('--rule_based_edi_mt_anno_form', action='store_const',
                         const='rule_based_edi_mt_anno_form', dest='task');  
    optParser.add_option('--pig_stream_collect_click_after_sort', action='store_const',
                         const='pig_stream_collect_click_after_sort', dest='task');     
    optParser.add_option('--pig_stream_get_possible_vertical_per_query', action='store_const',
                         const='pig_stream_get_possible_vertical_per_query', dest='task');                       
    optParser.add_option('--prep_data_for_infer', action='store_const',
                         const='prep_data_for_infer', dest='task');                       
    optParser.add_option('--pig_stream_gather_click_data_after_group', action='store_const',
                         const='pig_stream_gather_click_data_after_group', dest='task');                       
                
                
    (options, args) = optParser.parse_args();
    
    if(options.task == 'get_editor_annotation'):
        map = data_process.collectEditorialAnnotationMap;
        reduce = data_process.collectEditorialAnnotationReduce;
    elif(options.task == 'get_mt_annotation'):
        map = data_process.collectMTAnnotationMap;
        reduce = data_process.collectMTAnnotationReduce;
    elif(options.task == 'sort_mt_annotation'): data_process.sortMTResult;
    elif(options.task == 'get_ver_dist'):
        map = data_process.verDistMap;
        reduce = data_process.verDistReduce;
    elif(options.task == 'verDistDebug'): data_process.verDistDebug(args);
    elif(options.task == 'debug'):
#         map = data_process.getNqueryNumMap;
#         reduce = data_process.getNqueryNumReduce;
#         map = data_process.getQueryFeaLstMap;
#         map = data_process.checkFsStructureMap;
        data_process.avgFvPigStream(args);
    elif(options.task == 'get_nquery'): data_process.getNquery(args);
    elif(options.task == 'avg_fv'): data_process.avgFvPigStream(args);
    elif(options.task == 'sim_mt_merge_query'): simulate_annotation.mtAnnotationMergePerQuery(args);
    elif(options.task == 'extract_nquery_vertical'): simulate_annotation.extractNqueryVertical(args);
    elif(options.task == 'accum_nquery_vertical'): simulate_annotation.accumulateNqueryVertical(args);
    elif(options.task == 'get_editor_uniq_nquery'): simulate_annotation.getEditorUniqNquery(args);
    elif(options.task == 'get_trans_with_vertical'): simulate_annotation.getTransWithVertical(args);
    elif(options.task == 'rule_based_edi_mt_anno_form'): simulate_annotation.ruleBasedFormingEdiMtAnnotation(args);
    elif(options.task == 'pig_stream_collect_click_after_sort'): simulate_annotation.pigStreamCollectClickAfterSort(args);
    elif(options.task == 'prep_data_for_infer'): simulate_annotation.prepDataForInfer(args);
    elif(options.task == 'pig_stream_get_possible_vertical_per_query'): simulate_annotation.pigStreamGetPossibleVerticalPerQuery(args);
    elif(options.task == 'pig_stream_gather_click_data_after_group'): simulate_annotation.pigStreamGatherClickDataAfterGrouped(args);
    
    #===========================================================================
    # Map Reduce
    #===========================================================================
    if(options.m): map(args);
    elif(options.r): reduce(args);
    pass;