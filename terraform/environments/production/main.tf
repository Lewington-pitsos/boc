module "boc_production" {
  source = "../../main"
  instance_type = "g6e.2xlarge" # 20 Gbps bandwidth = 2500 MBps, 48 GPU ram, 16 VCPU
}