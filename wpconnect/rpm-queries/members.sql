select distinct
	  mem.*
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