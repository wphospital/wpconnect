select distinct
	  mem.id
	, mem.username
	, mem.email
	, mem.first_name
	, mem.last_name
	, mem.extern_id
	, case when
		mem.program = '' then 'None'
		else coalesce(mem.program, 'None')
		end as program
	, mem.birth_date
	, case when mem.extern_id ~ '^\d+$' then lpad(extern_id, 8, '0') else null end as mrn
	, bt.enroll_epoch
	, to_timestamp(bt.enroll_epoch) as enroll_datetime
	, to_timestamp(bt.enroll_epoch)::date as enroll_date
	, date_trunc('month', to_timestamp(bt.enroll_epoch)::date) as enroll_month
from rpm.members mem
inner join (
		select distinct member_id, enroll_epoch from rpm.billing_tally
	) bt on bt.member_id = mem.id
where mem.last_name not like '%%test%%'