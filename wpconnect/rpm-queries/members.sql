select distinct
	  mem.*
	, bt.enroll_epoch
	, to_timestamp(bt.enroll_epoch) as enroll_datetime
	, to_timestamp(bt.enroll_epoch)::date as enroll_date
from rpm.members mem
inner join rpm.billing_tally bt on bt.member_id = mem.id
where mem.last_name not like '%%test%%'