select
      bt.member_id
    , to_timestamp(bt.enroll_epoch) as enroll_date
    , mem.first_name
    , mem.last_name
    , mem.extern_id
    , mem.email
    , mem.first_name || ' ' || mem.last_name as full_name
    , bt.billing_period
    , bt.start_period
    , bt.end_period
    , bt.measurements_per_period
    , bt.setup_code
    , bt.monthly_code
    , bt.high_use_code
from rpm.billing_tally bt
left join rpm.members mem on mem.id = bt.member_id
where mem.last_name not like '%%test%%'
order by bt.end_period desc
