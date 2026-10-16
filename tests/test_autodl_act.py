"""Offline billing/credential/artifact tests. Explicit fixtures; no API, SSH or GPU."""
import contextlib
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('autodl_act', ROOT / 'autoDL/autodl_act.py')
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)
TEMP_ROOT = ROOT / 'temp/autodl-tests'


class AutoDLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        (TEMP_ROOT / 'README.md').write_text('Created/used: 2026-10-14 onward. Task: offline AutoDL scaffold tests. Test-owned subdirectories are removed automatically; this directory can be deleted when no tests are running. No credentials or real training results.\n', encoding='utf-8')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self):
        cfg = json.loads((ROOT / 'autoDL/configs/act.example.json').read_text())
        data = self.root / 'data'
        for folder in ('meta', 'data', 'videos'):
            (data / folder).mkdir(parents=True, exist_ok=True)
        (data / 'meta/info.json').write_text('{}')
        (data / 'data/fixture.txt').write_text('explicit test fixture')
        (data / 'videos/fixture.mp4').write_text('not a real video: no decoder runs in these tests')
        key = self.root / 'key'
        key.write_text('not a real key: SSH is mocked')
        Path(str(key) + '.pub').write_text('ssh-ed25519 test-public-key fixture')
        cfg.update(dataset_root=str(data), dataset_repo='test/data_v1', output_root=str(self.root / 'out'),
                   ssh_key=str(key), hourly_price_yuan=2.78, quote_checked_at=cloud.now(), run_name='fixture-run')
        return cfg

    def test_status_header_and_get_query(self):
        response = io.BytesIO(b'{"code":"Success","data":"shutdown"}')
        with patch.object(cloud, 'urlopen', return_value=response) as opened:
            self.assertEqual(cloud.Api('fake-token').call('status', {'instance_uuid':'pro-fixture'}), 'shutdown')
        request = opened.call_args.args[0]
        self.assertEqual(request.method, 'GET')
        self.assertIn('?instance_uuid=pro-fixture', request.full_url)
        self.assertEqual(request.headers['Authorization'], 'fake-token')

    def test_http_200_api_failure_rejected_and_not_retried(self):
        with patch.object(cloud, 'urlopen', return_value=io.BytesIO(b'{"code":"InsufficientBalance"}')) as net:
            with self.assertRaises(RuntimeError): cloud.Api('fake-token').call('create', {})
            self.assertEqual(net.call_count, 1)

    def test_wallet_uses_documented_post_and_milliyuan_fields(self):
        with patch.object(cloud,'urlopen',return_value=io.BytesIO(b'{"code":"Success","data":{"assets":10000}}')) as opened:
            wallet=cloud.Api('fixture-token').call('wallet',{})
        self.assertEqual(opened.call_args.args[0].method,'POST')
        self.assertEqual(wallet['assets']/cloud.PRICE_UNITS_PER_YUAN,10)

    def test_preflight_accepts_real_empty_private_image_response(self):
        api=Mock()
        api.instances.return_value=[]
        api.call.side_effect=[{'assets':50000,'remittance_code':'fixture-private-value'},
                              {'list':None,'page_index':1,'page_size':100,'max_page':1,'result_total':0}]
        output=io.StringIO()
        with patch.object(cloud,'read_token',return_value='fixture-token'),patch.object(cloud,'Api',return_value=api),contextlib.redirect_stdout(output):
            self.assertEqual(cloud.main(['preflight']),0)
        result=json.loads(output.getvalue())
        self.assertEqual(result['private_images'],[])
        self.assertEqual(result['instances'],[])
        self.assertEqual(result['wallet']['assets_yuan'],50)
        self.assertNotIn('fixture-private-value',output.getvalue())
        self.assertNotIn('fixture-token',output.getvalue())
        self.assertEqual([c.args[0] for c in api.call.call_args_list],['wallet','image/private/list'])

    def test_empty_instance_null_page_is_safe_but_ambiguous_pages_fail(self):
        api=cloud.Api('fixture-token')
        api.call=Mock(return_value={'list':None,'result_total':0,'max_page':1})
        self.assertEqual(api.instances(),[])
        for response in ({'list':None,'result_total':1}, {'list':None}, {}, None,
                         {'list':[],'result_total':1}, {'list':[{}],'result_total':0},
                         {'list':'invalid','result_total':0}):
            with self.subTest(response=response),self.assertRaises(ValueError):
                cloud.page_items(response,'fixture-list')

    def test_read_token_file_and_explicit_file_precedence(self):
        file = self.root / 'token'; file.write_text('fixture-token\n'); file.chmod(0o600)
        with patch.dict(os.environ, {'AUTODL_TOKEN':'other-fixture-token'}):
            self.assertEqual(cloud.read_token(file), 'fixture-token')
            self.assertEqual(cloud.read_token(), 'other-fixture-token')
            with self.assertRaises(FileNotFoundError): cloud.read_token(self.root / 'missing')

    @unittest.skipIf(os.name == 'nt', 'POSIX mode check')
    def test_publicly_readable_token_refused(self):
        file = self.root / 'token'; file.write_text('fixture-token'); file.chmod(0o644)
        with self.assertRaises(PermissionError): cloud.read_token(file)

    def test_auth_is_hidden_and_refuses_overwrite(self):
        file = self.root / 'secrets/token'
        output = io.StringIO()
        with patch.object(cloud.getpass, 'getpass', return_value='fixture-token'), contextlib.redirect_stdout(output):
            cloud.configure_token(file)
            with self.assertRaises(FileExistsError): cloud.configure_token(file)
        self.assertNotIn('fixture-token', output.getvalue())
        if os.name != 'nt': self.assertEqual(file.stat().st_mode & 0o777, 0o600)

    def test_dry_run_uses_no_token_network_ssh_or_gpu(self):
        with patch.object(cloud, 'read_token') as token, patch.object(cloud, 'urlopen') as net, patch.object(cloud.subprocess, 'run') as shell:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.main(['submit','--config',str(ROOT/'autoDL/configs/act.example.json'),'--dry-run']),0)
            token.assert_not_called(); net.assert_not_called(); shell.assert_not_called()

    def test_quote_required_stale_quote_and_nan_budget_rejected(self):
        cfg = self.config(); file=self.root/'config.json'
        for changes in ({'hourly_price_yuan':None},{'quote_checked_at':'2020-01-01T00:00:00+00:00'},{'max_cost_yuan':float('nan')}):
            file.write_text(json.dumps({**cfg,**changes}))
            with self.subTest(changes=changes), self.assertRaises(ValueError): cloud.load_config(file,paid=True)

    def test_budget_counts_setup_and_reserves_retrieval_and_shutdown(self):
        cfg=self.config(); seconds=cloud.budget_seconds(cfg)
        self.assertLessEqual(seconds/3600*cfg['hourly_price_yuan']+cfg['storage_reserve_yuan'],cfg['max_cost_yuan'])
        cfg['retrieve_minutes']=seconds//60+1
        with self.assertRaises(ValueError): cloud.budget_seconds(cfg)

    def test_secret_configuration_keys_refused(self):
        file=self.root/'config.json';file.write_text(json.dumps({**self.config(),'token':'fixture-secret'}))
        with self.assertRaises(ValueError):cloud.load_config(file)

    def test_manifest_excludes_private_reports_and_rejects_symlinks(self):
        cfg=self.config();root=Path(cfg['dataset_root']);(root/'private-report.txt').write_text('private fixture')
        self.assertNotIn('private-report.txt',cloud.dataset_manifest(root))
        if os.name!='nt':
            (root/'data/leak').symlink_to(root/'private-report.txt')
            with self.assertRaises(ValueError):cloud.dataset_manifest(root)

    def test_download_change_and_escape_rejected(self):
        bundle=self.root/'bundle';bundle.mkdir();file=bundle/'result.json';file.write_text('{"completed":false}')
        cloud.write_json(bundle/'sha256.json',{'result.json':cloud.sha256(file)})
        self.assertFalse(cloud.verify_bundle(bundle)['completed'])
        file.write_text('changed')
        with self.assertRaises(ValueError):cloud.verify_bundle(bundle)
        outside=self.root/'outside';outside.write_text('fixture')
        cloud.write_json(bundle/'sha256.json',{'../outside':cloud.sha256(outside)})
        with self.assertRaises(ValueError):cloud.verify_bundle(bundle)

    def test_failed_status_read_still_power_off_and_observed_state_required(self):
        api=cloud.Api('fixture-token');api.call=Mock(side_effect=[OSError('status unavailable'),None,'stopping','shutdown'])
        with patch.object(cloud.time,'sleep'):self.assertEqual(api.stop('pro-fixture'),'shutdown')
        self.assertEqual(api.call.call_args_list[1].args[0],'power_off')

    def test_stop_timeout_does_not_claim_success(self):
        api=cloud.Api('fixture-token');api.call=Mock(return_value='running')
        with patch.object(cloud.time,'monotonic',side_effect=[0,100]), self.assertRaises(TimeoutError):api.stop('pro-fixture')

    def test_release_requires_stopped_state_and_observed_disappearance(self):
        api=cloud.Api('fixture-token');api.call=Mock(return_value='running')
        with self.assertRaises(ValueError):api.release('pro-fixture')
        api.call=Mock(side_effect=['shutdown',None]);api.instances=Mock(side_effect=[[{'uuid':'pro-fixture'}],[]])
        with patch.object(cloud.time,'sleep'):api.release('pro-fixture')
        self.assertEqual(api.instances.call_count,2)

    def api(self):
        api=Mock()
        api.call.side_effect=lambda op,payload:({'assets':10000} if op=='wallet' else 'pro-fixture' if op=='create' else {'proxy_host':'fixture.autodl.com','ssh_port':2222,'root_password':'MUST-NOT-PERSIST','jupyter_token':'ALSO-SECRET','payg_price':2780})
        api.stop.return_value='shutdown'
        api.instances.return_value=[]
        return api

    def transport(self,cfg,events,*,fail=None,completed=True):
        def execute(command,**kwargs):
            if fail and fail(command):raise OSError('fixture transport failure')
            if command[0]=='scp' and any('/result/.' in str(v) for v in command):
                events.append('download')
                dest=Path(cfg['output_root'])/cfg['run_name']/'result'
                cloud.write_json(dest/'result.json',{'completed':completed,'exit_code':0 if completed else 1})
                cloud.write_json(dest/'sha256.json',{'result.json':cloud.sha256(dest/'result.json')})
            return subprocess.CompletedProcess(command,0)
        return execute

    def run_submit(self,cfg,api,transport):
        with patch.object(cloud.shutil,'which',return_value='/test/ssh'),patch.object(cloud.subprocess,'run',side_effect=transport),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return cloud.submit(cfg,api)

    def test_success_download_precedes_stop_then_release_no_secrets(self):
        cfg=self.config();api=self.api();events=[]
        api.stop.side_effect=lambda instance:events.append('stop') or 'shutdown'
        api.release.side_effect=lambda instance:events.append('release')
        self.assertEqual(self.run_submit(cfg,api,self.transport(cfg,events)),0)
        self.assertEqual(events,['download','stop','release'])
        receipt=(Path(cfg['output_root'])/cfg['run_name']/'controller.json').read_text()
        self.assertNotIn('MUST-NOT-PERSIST',receipt);self.assertNotIn('ALSO-SECRET',receipt)
        self.assertTrue(json.loads(receipt)['release_verified'])
        self.assertEqual(sum(c.args[0]=='create' for c in api.call.call_args_list),1)

    def test_setup_failure_stops_and_releases_new_untrained_instance(self):
        cfg=self.config();api=self.api()
        transport=self.transport(cfg,[],fail=lambda c:c[0]=='scp')
        self.assertEqual(self.run_submit(cfg,api,transport),1)
        api.stop.assert_called_once_with('pro-fixture');api.release.assert_called_once_with('pro-fixture')

    def test_actual_price_above_quote_stops_before_ssh_or_setup(self):
        cfg=self.config();cfg['hourly_price_yuan']=1;api=self.api();transport=Mock()
        self.assertEqual(self.run_submit(cfg,api,transport),1)
        transport.assert_not_called();api.stop.assert_called_once();api.release.assert_called_once()

    def test_insufficient_cash_balance_never_creates(self):
        cfg=self.config();api=self.api();api.call.return_value={'assets':9999};api.call.side_effect=None
        with self.assertRaises(ValueError):self.run_submit(cfg,api,Mock())
        self.assertFalse(any(c.args[0]=='create' for c in api.call.call_args_list))

    def test_unfinished_private_image_never_creates(self):
        cfg=self.config();cfg['image_uuid']='image-fixture';api=self.api()
        api.call.side_effect=lambda op,payload:{'assets':10000} if op=='wallet' else {'list':[{'image_uuid':'image-fixture','status':'saving'}]}
        with self.assertRaises(ValueError):self.run_submit(cfg,api,Mock())
        self.assertFalse(any(c.args[0]=='create' for c in api.call.call_args_list))

    def test_token_environment_is_not_passed_to_ssh_or_scp(self):
        cfg=self.config();api=self.api();environments=[];base=self.transport(cfg,[])
        def transport(command,**kwargs):
            environments.append(kwargs['env']);return base(command,**kwargs)
        with patch.dict(os.environ,{'AUTODL_TOKEN':'fixture-secret','AUTODL_PRO_TOKEN':'other-fixture-secret'}):
            self.assertEqual(self.run_submit(cfg,api,transport),0)
        self.assertTrue(environments)
        self.assertTrue(all('AUTODL_TOKEN' not in e and 'AUTODL_PRO_TOKEN' not in e for e in environments))

    def test_download_failure_retains_unique_results_but_stops_compute(self):
        cfg=self.config();api=self.api()
        transport=self.transport(cfg,[],fail=lambda c:any('/result/.' in str(v) for v in c))
        self.assertEqual(self.run_submit(cfg,api,transport),1)
        api.stop.assert_called_once();api.release.assert_not_called()
        receipt=json.loads((Path(cfg['output_root'])/cfg['run_name']/'controller.json').read_text())
        self.assertFalse(receipt['download_verified']);self.assertIn('storage_retained_reason',receipt)

    def test_keyboard_interrupt_and_timeout_both_stop(self):
        for error in (KeyboardInterrupt(),subprocess.TimeoutExpired('fixture-ssh',1)):
            with self.subTest(error=type(error).__name__):
                cfg=self.config();cfg['run_name']=type(error).__name__;api=self.api()
                def transport(command,**kwargs):raise error
                self.assertEqual(self.run_submit(cfg,api,transport),1);api.stop.assert_called_once()

    def test_failed_cleanup_is_recorded(self):
        cfg=self.config();api=self.api();api.stop.side_effect=TimeoutError('fixture stop unconfirmed')
        self.assertEqual(self.run_submit(cfg,api,self.transport(cfg,[])),1)
        receipt=json.loads((Path(cfg['output_root'])/cfg['run_name']/'controller.json').read_text())
        self.assertFalse(receipt['shutdown_verified']);self.assertFalse(receipt['release_verified'])

    def test_unknown_creation_recovers_only_owned_name_without_recreating(self):
        cfg=self.config();api=self.api()
        def call(op,payload):
            if op=='wallet':return {'assets':10000}
            if op=='create':
                api.instances.return_value=[{'name':payload['instance_name'],'uuid':'pro-fixture'}]
                raise TimeoutError('fixture lost response')
            self.fail('No further non-cleanup call after unknown create')
        api.call.side_effect=call
        self.assertEqual(self.run_submit(cfg,api,Mock()),1)
        self.assertEqual(sum(c.args[0]=='create' for c in api.call.call_args_list),1);api.stop.assert_called_once();api.release.assert_called_once()

    def test_existing_output_refuses_all_api_calls(self):
        cfg=self.config();(Path(cfg['output_root'])/cfg['run_name']).mkdir(parents=True);api=self.api()
        with self.assertRaises(FileExistsError):self.run_submit(cfg,api,Mock())
        self.assertFalse(any(c.args[0]=='create' for c in api.call.call_args_list))

    def test_start_command_contains_no_api_secret_and_watchdog_is_detached(self):
        command=cloud.startup_command(self.config(),'/root/test-run','ssh-ed25519 public-fixture',12345)
        source=shlex.split(command)[2]
        compile(source,'startup','exec')
        self.assertIn('start_new_session=True',source);self.assertIn('watchdog-ready.json',source)
        self.assertNotIn('AUTODL_TOKEN',source)
        constants = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Constant)]
        self.assertIn('\n', constants)
        self.assertNotIn('\\n', constants)

    def test_cuda_index_versions_accepted_without_accepting_other_builds(self):
        self.assertIn('2.11.0+cu130',cloud.accepted_versions('torch'))
        self.assertIn('0.26.0+cu130',cloud.accepted_versions('torchvision'))
        self.assertNotIn('2.11.0+cpu',cloud.accepted_versions('torch'))
        self.assertNotIn('2.11.0+cu128',cloud.accepted_versions('torch'))
        source=shlex.split(cloud.bootstrap_commands(self.config(),'/root/test').split('; then')[0].split('if ! ',1)[1])[2]
        compile(source,'version-check','exec')
        with patch.object(sys,'version_info',(3,12)),patch.object(cloud.importlib.metadata,'version',side_effect=lambda n:cloud.EXPECTED_PACKAGES[n]+('+cu130' if n in {'torch','torchvision'} else '')):
            exec(source)

    def checkpoint(self,path,step,repo='test/data_v1'):
        model=path/'pretrained_model';state=path/'training_state';model.mkdir(parents=True,exist_ok=True);state.mkdir()
        for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):(model/name).write_text('{}' if name.endswith('.json') else 'explicit fixture')
        cloud.write_json(model/'train_config.json',{'dataset':{'repo_id':repo,'root':'fixture'},'policy':{},'wandb':{},'steps':step,'output_dir':'fixture','save_freq':20000})
        cloud.write_json(state/'training_step.json',{'step':step})
        for name in ('optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors'):(state/name).write_text('{}' if name.endswith('.json') else 'explicit fixture')

    def test_resume_command_preserves_original_and_rewrites_new_config(self):
        source=self.root/'060000';self.checkpoint(source,60000)
        args=cloud.parser().parse_args(['worker','--dataset-root',str(self.root),'--dataset-repo','test/data_v1','--run-name','fixture','--max-minutes','240','--expected-data','fixture','--train-seconds','10000','--resume-checkpoint',str(source)])
        bundle=self.root/'bundle';bundle.mkdir();command=cloud.training_command(args,bundle)
        self.assertIn('--resume=true',command)
        copied=bundle/'train/checkpoints/060000/pretrained_model/train_config.json'
        self.assertEqual(json.loads(copied.read_text())['steps'],100000)
        self.assertEqual(json.loads((source/'pretrained_model/train_config.json').read_text())['steps'],60000)

    def test_worker_resume_requires_only_new_save_points(self, missing_state=False):
        cfg=self.config();source=self.root/'060000';self.checkpoint(source,60000)
        expected=self.root/'expected.json';cloud.write_json(expected,cloud.dataset_manifest(Path(cfg['dataset_root'])))
        args=cloud.parser().parse_args(['worker','--dataset-root',cfg['dataset_root'],'--dataset-repo',cfg['dataset_repo'],'--run-name',str(self.root/'worker-result'),'--max-minutes','240','--expected-data',str(expected),'--train-seconds','10000','--resume-checkpoint',str(source)])
        def execute(command,**kwargs):
            if '-c' in command:return subprocess.CompletedProcess(command,0,stdout='{"gpu":"fixture GPU","cuda":"13.0"}')
            bundle=Path(args.run_name)
            for step in (80000,100000):self.checkpoint(bundle/f'train/checkpoints/{step:06d}',step)
            if missing_state:
                (bundle/'train/checkpoints/080000/training_state/optimizer_state.safetensors').unlink()
            last=bundle/'train/checkpoints/last'
            if os.name=='nt':
                import shutil
                shutil.copytree(bundle/'train/checkpoints/100000',last)
            else:last.symlink_to('100000',target_is_directory=True)
            return subprocess.CompletedProcess(command,0)
        with patch.object(cloud.importlib.metadata,'version',side_effect=lambda n:cloud.EXPECTED_PACKAGES[n]),patch.object(cloud.subprocess,'run',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cloud.worker(args)==0,not missing_state)
        self.assertEqual(json.loads((Path(args.run_name)/'result.json').read_text())['completed'],not missing_state)
        self.assertFalse((Path(args.run_name)/'train/checkpoints/020000').exists())

    def test_missing_optimizer_at_intermediate_checkpoint_is_incomplete(self):
        self.test_worker_resume_requires_only_new_save_points(missing_state=True)

    def test_weight_validation_failure_and_timeout_cannot_report_completed(self):
        for error in (subprocess.CalledProcessError(1,'fixture-finite-check'),subprocess.TimeoutExpired('fixture-finite-check',1)):
            cfg=self.config();tag=type(error).__name__;source=self.root/('source-'+tag);self.checkpoint(source,60000)
            expected=self.root/('expected-'+tag+'.json');cloud.write_json(expected,cloud.dataset_manifest(Path(cfg['dataset_root'])))
            args=cloud.parser().parse_args(['worker','--dataset-root',cfg['dataset_root'],'--dataset-repo',cfg['dataset_repo'],'--run-name',str(self.root/tag),'--max-minutes','240','--expected-data',str(expected),'--train-seconds','10000','--resume-checkpoint',str(source)])
            def execute(command,**kwargs):
                if '-c' in command:
                    if 'load_file' in command[2]:raise error
                    return subprocess.CompletedProcess(command,0,stdout='{"gpu":"fixture GPU","cuda":"13.0"}')
                for step in (80000,100000):self.checkpoint(Path(args.run_name)/f'train/checkpoints/{step:06d}',step)
                last=Path(args.run_name)/'train/checkpoints/last'
                import shutil
                shutil.copytree(last.with_name('100000'),last)
                return subprocess.CompletedProcess(command,0)
            with self.subTest(error=tag),patch.object(cloud.importlib.metadata,'version',side_effect=lambda n:cloud.EXPECTED_PACKAGES[n]),patch.object(cloud.subprocess,'run',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(cloud.worker(args),0)
            self.assertFalse(json.loads((Path(args.run_name)/'result.json').read_text())['completed'])

    def test_health_resume_and_budget_abort_both_keep_real_fixture_checkpoints(self):
        for abort in (False,True):
            cfg=self.config();tag='budget-abort' if abort else 'health-resume';expected=self.root/(tag+'.json')
            cloud.write_json(expected,cloud.dataset_manifest(Path(cfg['dataset_root'])))
            args=cloud.parser().parse_args(['worker','--dataset-root',cfg['dataset_root'],'--dataset-repo',cfg['dataset_repo'],'--run-name',str(self.root/tag),'--max-minutes','240','--expected-data',str(expected),'--train-seconds','100' if abort else '10000','--steps','100000' if abort else '2000','--save-freq','500'])
            clock=[0.0]
            def execute(command,**kwargs):
                if '-c' in command:return subprocess.CompletedProcess(command,0,stdout='{"gpu":"fixture GPU","cuda":"13.0"}')
                if any(c.startswith('--config_path=') for c in command):
                    train=json.loads(Path(next(c.split('=',1)[1] for c in command if c.startswith('--config_path='))).read_text());out=Path(train['output_dir']);steps=[1500,2000]
                else:
                    out=Path(next(c.split('=',1)[1] for c in command if c.startswith('--output_dir=')));steps=[1000];clock[0]=2.0
                kwargs['stdout'].write('INFO loss:0.5 l1_loss:0.4 kld_loss:0.01\n');kwargs['stdout'].flush()
                for step in steps:self.checkpoint(out/f'checkpoints/{step:06d}',step)
                import shutil
                shutil.copytree(out/f'checkpoints/{steps[-1]:06d}',out/'checkpoints/last')
                return subprocess.CompletedProcess(command,0)
            with self.subTest(abort=abort),patch.object(cloud.time,'monotonic',side_effect=lambda:clock[0]),patch.object(cloud.importlib.metadata,'version',side_effect=lambda n:cloud.EXPECTED_PACKAGES[n]),patch.object(cloud.subprocess,'run',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.worker(args)==0,not abort)
            self.assertTrue((Path(args.run_name)/'health/train/checkpoints/001000/training_state/training_step.json').exists())
            self.assertFalse((Path(args.run_name)/'train/checkpoints/000500').exists())


if __name__=='__main__':unittest.main()
